import asyncio
import json
import logging
import numpy as np
from scipy.fft import fft
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import AnyUrl, Field
import tinytuya
import subprocess
import os
import inspect

DEVICES_FILE = os.environ.get('DEVICES', os.path.expanduser('~/snapshot.json'))
XDG_RUNTIME_DIR = os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
AUDIO_TARGET = os.environ.get('TUYA_MCP_AUDIO_TARGET', 'alsa_output.pci-0000_00_1b.0.analog-stereo.monitor')

devices = []
mcp = FastMCP("tuya_mcp_server")

# Load devices from the snapshot file
def load_devices() -> dict:
    global devices
    try:
        with open(DEVICES_FILE, 'r') as f:
            devices = json.load(f).get("devices", [])
        return {"status": "success"}
    except FileNotFoundError:
        logging.error(f"Error: {DEVICES_FILE} not found.")
        return {"status": "error", "message": f"Error: {DEVICES_FILE} not found."}
    except json.JSONDecodeError:
        logging.error(f"Error: Invalid JSON format in {DEVICES_FILE}.")
        return {"status": "error", "message": f"Error: Invalid JSON format in {DEVICES_FILE}."}

# Parse audio data into HSV values
def parse_audio(audio_data, freq_range) -> tuple:
    sample_rate = 44100
    n = len(audio_data)
    if n == 0:
        return (0, 0, 0)

    fft_data = np.fft.fft(audio_data)
    fft_abs = np.abs(fft_data[:n // 2])
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    freq_mask = np.logical_and(frequencies >= freq_range[0], frequencies <= freq_range[1])
    freq_range_values = fft_abs[freq_mask]

    if not freq_range_values.size:
        return (0, 0, 0)

    dominant_frequency_index = np.argmax(freq_range_values)
    dominant_frequency = frequencies[freq_mask][dominant_frequency_index]
    amplitude = freq_range_values[dominant_frequency_index]

    hue = (dominant_frequency / freq_range[1]) * 360
    saturation = min(amplitude / 10000, 1.0) * 1000
    value = min(amplitude / 5000, 1.0) * 1000

    return hue, saturation, value

# Generic tool for controlling Tuya devices
@mcp.tool()
async def control_device(
    device_id: str = Field(description="The Tuya device ID."),
    local_key: str = Field(description="The key to control the device."),
    ip_address: str = Field(description="The IP address of the device."),
    version: str = Field(description="The version of the device."),
    device_type: str = Field(description="The type of the device (e.g., 'Device', 'BulbDevice').", default="Device"),
    action: str = Field(description="The method to call on the device."),
    args: list = Field(description="Positional arguments for the method.", default=[]),
    kwargs: dict = Field(description="Keyword arguments for the method.", default={}),
) -> str:
    """Generic tool to call any method on a Tuya device."""
    try:
        # Get the device class from tinytuya
        device_class = getattr(tinytuya, device_type, None)
        if not device_class:
            return f"Error: Device type '{device_type}' not found."

        # Create the device instance
        device = device_class(device_id, ip_address, local_key)
        device.set_version(version)

        # Get the method to call
        method = getattr(device, action, None)
        if not method:
            return f"Error: Method '{action}' not found on device type '{device_type}'."

        # Call the method
        result = await asyncio.to_thread(method, *args, **kwargs)
        return f"Action '{action}' executed successfully on device '{device_id}'. Result: {result}"

    except Exception as e:
        logging.error(f"Error controlling device {device_id}: {e}")
        return f"Error: {e}"

# Dynamically create tools for all methods of a device class
def create_tools_for_device_class(device_class):
    for name, method in inspect.getmembers(device_class, inspect.isfunction):
        if not name.startswith("_"):  # Skip private methods
            @mcp.tool(name=f"{device_class.__name__}.{name}")
            async def tool(
                device_id: str = Field(description="The Tuya device ID."),
                local_key: str = Field(description="The key to control the device."),
                ip_address: str = Field(description="The IP address of the device."),
                version: str = Field(description="The version of the device."),
                *args,
                **kwargs,
            ) -> str:
                return await control_device(
                    device_id=device_id,
                    local_key=local_key,
                    ip_address=ip_address,
                    version=version,
                    device_type=device_class.__name__,
                    action=name,
                    args=args,
                    kwargs=kwargs,
                )

# Create tools for all device classes in tinytuya
for name, device_class in inspect.getmembers(tinytuya, inspect.isclass):
    if name.endswith("Device"):  # Only process device classes
        create_tools_for_device_class(device_class)

# Music mode tool
@mcp.tool()
async def set_music_mode(device_name: str, delay: float = 0.01) -> str:
    all_devices = device_name == 'all'
    min_freq = 100
    max_freq = 2000
    num_devices = len(devices)

    if num_devices == 0:
        raise ValueError("No devices available.")

    step = (max_freq - min_freq) // num_devices
    frequency_ranges = [(min_freq + i * step, min_freq + (i + 1) * step) for i in range(num_devices)]
    started_devices = []

    for i, device in enumerate(devices):
        if device.get('name') == device_name or all_devices:
            freq_range = frequency_ranges[i]
            try:
                process = await asyncio.create_subprocess_exec(
                    'parec', '-d', AUDIO_TARGET, '--format=s16le', '--rate=44100',
                    env={'XDG_RUNTIME_DIR': XDG_RUNTIME_DIR},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                async def audio_process_async(d, process, delay, freq_range):
                    while True:
                        stdout = await process.stdout.read(44100)
                        audio_data = np.frombuffer(stdout, dtype=np.int16).astype(float)
                        hue, saturation, value = parse_audio(audio_data, freq_range)
                        await control_device(
                            device_id=d['id'],
                            local_key=d['key'],
                            ip_address=d['ip'],
                            version=d['ver'],
                            action="set_hsv",
                            args=[hue / 360.0, saturation / 1000.0, value / 1000.0],
                        )

                asyncio.create_task(audio_process_async(device, process, delay, freq_range))
                started_devices.append(f"{device['name']} ({freq_range[0]}-{freq_range[1]} Hz)")
            except Exception as e:
                raise ValueError(f"Error setting color for device {device_name}: {e}")

    if started_devices:
        return f"Music mode started for {', '.join(started_devices)}"
    raise ValueError("No matching devices found")

# Resource for listing device capabilities
class DeviceCapabilitiesResource(types.Resource):
    async def read(self):
        device_data = next((device for device in devices if device.get("name") == self.name), None)
        if not device_data:
            raise ValueError(f"Device '{self.name}' not found in devices list")

        device_class = getattr(tinytuya, device_data.get("type", "Device"), tinytuya.Device)
        capabilities = [name for name, _ in inspect.getmembers(device_class, inspect.isfunction) if not name.startswith("_")]

        return json.dumps({
            "uri": str(self.uri),
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "capabilities": capabilities,
            "data": { k: v for k, v in device_data.items() if k != "key" }
        })

# Create resources for devices
def make_devices() -> list[DeviceCapabilitiesResource]:
    return [
        DeviceCapabilitiesResource(
            uri=AnyUrl(f"tuya://device/{device['name']}"),
            name=device['name'],
            description=f"A Tuya device named {device['name']}",
            mime_type="application/json",
        )
        for device in devices
    ]

# Main entry point
def main():
    """Main entry point for the server."""
    if load_devices().get("status") != "success":
        return

    for resource in make_devices():
        mcp.add_resource(resource)

    # Run the MCP server
    mcp.run()
