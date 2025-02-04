import logging
import asyncio
import json
import time
import subprocess
import numpy as np
from scipy.fft import fft

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
import tinytuya
import sys
import os
import mcp.server.stdio
import queue
import threading

DEVICES_FILE = os.environ.get('TUYA_MCP_DEVICES_FILE', os.path.expanduser('~/snapshot.json'))
SINKORSOURCE = os.environ.get('TUYA_MCP_AUDIO_MONITOR', 'alsa_output.pci-0000_00_1b.0.analog-stereo.monitor')
SAMPLE_RATE  = os.environ.get('TUYA_MCP_AUDIO_SAMPLE_RATE', 48000)
XDG_RUNTIME_DIR = os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.geteuid()}')

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

devices = []
music_processes = {}
audio_buffer = queue.Queue()

server = Server("tuya_mcp_server")

def load_devices():
    global devices
    try:
        with open(DEVICES_FILE, 'r') as f:
            devices = json.load(f).get("devices", [])
        return {"status": "success"}
    except FileNotFoundError:
        print(f"Error: {DEVICES_FILE} not found.")
        return {"status": "error", "message": f"Error: {DEVICES_FILE} not found."}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {DEVICES_FILE}.")
        return {"status": "error", "message": f"Error: Invalid JSON format in {DEVICES_FILE}."}

def parse_audio(audio_data, sample_rate, freq_range):
    n = len(audio_data)
    if n == 0:
        return (0, 0, 0)

    fft_data = np.fft.fft(audio_data)
    fft_abs = np.abs(fft_data[:n // 2])
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    # Extract only the assigned frequency range
    freq_mask = np.logical_and(frequencies >= freq_range[0], frequencies <= freq_range[1])
    freq_range_values = fft_abs[freq_mask]

    if not freq_range_values.size:
        return (0, 0, 0)

    # Find dominant frequency in the assigned range
    dominant_frequency_index = np.argmax(freq_range_values)
    dominant_frequency = frequencies[freq_mask][dominant_frequency_index]
    amplitude = freq_range_values[dominant_frequency_index]

    # Convert frequency and amplitude into HSV values
    hue = (dominant_frequency / freq_range[1]) * 360  
    saturation = min(amplitude / 10000, 1.0) * 1000  
    value = min(amplitude / 5000, 1.0) * 1000  

    return hue, saturation, value


color_names = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}

def parse_color(color_input):
    if isinstance(color_input, tuple) and len(color_input) == 3:
      return color_input
    if color_input.startswith('rgb(') and color_input.endswith(')'):
        try:
            parts = color_input[4:-1].split(',')
            r = int(parts[0].strip())
            g = int(parts[1].strip())
            b = int(parts[2].strip())
            return (r, g, b)
        except:
          return None
    elif color_input.startswith('#') and len(color_input) == 7:
        try:
            r = int(color_input[1:3], 16)
            g = int(color_input[3:5], 16)
            b = int(color_input[5:7], 16)
            return (r, g, b)
        except:
            return None
    elif color_input.lower() in color_names:
        return color_names[color_input.lower()]
    return None

async def control_device(device, action, *args, function_name='', **kwargs):
    try:
        device_id = device['id']
        local_key = device['key']
        ip_address = device['ip']
        version = device["ver"]

        d = tinytuya.BulbDevice(device_id, ip_address, local_key)
        d.set_version(version)
        d.set_socketPersistent(True)

        function = getattr(d, function_name or action)
        await asyncio.to_thread(function, *args, **kwargs)

    except Exception as e:
        print(f"Error controlling device {device.get('name', 'unknown')}: {e}")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available Tuya device resources.
    Each device is exposed as a resource with a custom tuya:// URI scheme.
    """
    return [
        types.Resource(
            uri=AnyUrl(f"tuya://device/{device['name']}"),
            name=f"Tuya Device: {device['name']}",
            description=f"A Tuya device named {device['name']}",
            mimeType="application/json",
        )
        for device in devices
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific device's information by its URI.
    The device name is extracted from the URI path.
    """
    if uri.scheme != "tuya":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    device_name = uri.path
    if device_name is not None:
        device_name = device_name.lstrip("/")
        for device in devices:
            if device["name"] == device_name:
                filtered_device = {k: v for k, v in device.items() if k not in [ "key" ]}
                return json.dumps(filtered_device)
    raise ValueError(f"Device not found: {device_name}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools for controlling Tuya devices.
    """
    return [
        types.Tool(
            name="turn_on",
            description="Turn on a Tuya device",
            inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "all": {"type": "boolean"}
                 },
            },
        ),
        types.Tool(
            name="turn_off",
            description="Turn off a Tuya device",
             inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                     "all": {"type": "boolean"}
                 },
            },
        ),
         types.Tool(
            name="set_color",
            description="Set the color of a Tuya device",
            inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "color": {"type": "string"},
                    "all": {"type": "boolean"}
                 },
                 "required": ["color"]
            },
        ),
        types.Tool(
            name="set_brightness",
            description="Set the brightness of a Tuya device",
             inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "brightness": {"type": "integer"},
                     "all": {"type": "boolean"}
                 },
                  "required": ["brightness"]
            },
        ),
        types.Tool(
            name="set_colourtemp",
            description="Set the colour temperature of a Tuya device",
             inputSchema={
                "type": "object",
                "properties": {
                     "device": {"type": "string"},
                    "colourtemp": {"type": "integer"},
                    "all": {"type": "boolean"}
                 },
                 "required": ["colourtemp"]
            },
        ),
        types.Tool(
            name="set_mode",
            description="Set the mode of a Tuya device. For 'music' mode you should try the tool called music.",
             inputSchema={
                "type": "object",
                "properties": {
                     "device": {"type": "string"},
                    "mode": {"type": "string", "enum": ["white", "colour", "scene", "music"]},
                    "all": {"type": "boolean"}
                 },
                "required": ["mode"]
            },
        ),
        types.Tool(
            name="music",
            description="Start music mode for a Tuya device",
             inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "delay": {"type": "number"},
                    "all": {"type": "boolean"}
                 },
            },
        ),
         types.Tool(
            name="stop_music",
            description="Stop music mode for a Tuya device",
             inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "all": {"type": "boolean"}
                 },
                 "required": ["device"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests for Tuya devices.
    Tools can modify server state and notify clients of changes.
    """
    if not arguments:
        raise ValueError("Missing arguments")
    if name == "turn_on":
        return await handle_action_over_devices(name, arguments)
    if name == "turn_off":
         return await handle_action_over_devices(name, arguments)
    if name == "set_color":
        return await handle_set_color(arguments)
    if name == "set_brightness":
         return await handle_set_brightness(arguments)
    if name == "set_colourtemp":
        return await handle_set_colourtemp(arguments)
    if name == "set_mode":
         return await handle_set_mode(arguments)
    if name == "music":
         return await handle_music(arguments)
    if name == "stop_music":
        return await handle_stop_music(arguments)

    raise ValueError(f"Unknown tool: {name}")

async def handle_action_over_devices(action, arguments):
    all_devices = arguments.get('all', False)
    device_name = arguments.get('device')

    if not all_devices and not device_name:
        raise ValueError("Device name must be provided")

    tasks = [
        control_device(device, action)
        for device in devices
        if device.get('name') == device_name or all_devices
    ]

    if tasks:
        await asyncio.gather(*tasks)
        if all_devices:
            return [types.TextContent(type="text", text=f"All devices {action}.")]
        return [types.TextContent(type="text", text=f"Device {device_name} {action}.")]

    raise ValueError(f"Device {device_name} not found")

async def handle_set_color(arguments):
    device_name = arguments.get('device')
    all_devices = arguments.get('all', False)
    color_input = arguments.get('color')

    if not color_input:
        raise ValueError("Color must be provided")
    
    rgb = parse_color(color_input)
    if not rgb or any(x is None for x in rgb):
        raise ValueError("Invalid color format")

    tasks = [
        control_device(device, "set_colour", *rgb)
        for device in devices
        if device.get('name') == device_name or all_devices
    ]

    if tasks:
        await asyncio.gather(*tasks)
        if all_devices:
            return [types.TextContent(type="text", text=f"All devices color set to {color_input}.")]
        return [types.TextContent(type="text", text=f"Device {device_name} color set to {color_input}.")]

    raise ValueError(f"Device {device_name} not found")

async def handle_set_brightness(arguments):
    all_devices = arguments.get('all', False)
    device_name = arguments.get('device')
    brightness = arguments.get('brightness')

    if (not all_devices and not device_name) or brightness is None:
        raise ValueError("Device and brightness must be provided")
    
    try:
        brightness = int(brightness)
    except ValueError:
        raise ValueError("Brightness must be an integer")
    
    if not (0 <= brightness <= 1000):
        raise ValueError("Brightness must be between 0 and 1000")

    tasks = [
        control_device(device, "set_brightness", brightness)
        for device in devices
        if device.get('name') == device_name or all_devices
    ]

    if tasks:
        await asyncio.gather(*tasks)
        if all_devices:
            return [types.TextContent(type="text", text=f"All devices brightness set to {brightness}.")]
        return [types.TextContent(type="text", text=f"Device {device_name} brightness set to {brightness}.")]

    raise ValueError(f"Device {device_name} not found")

async def handle_set_colourtemp(arguments):
    all_devices = arguments.get('all', False)
    device_name = arguments.get('device')
    colourtemp = arguments.get('colourtemp')

    if (not all_devices and not device_name) or colourtemp is None:
        raise ValueError("Device and colour temperature must be provided")

    try:
        colourtemp = int(colourtemp)
    except ValueError:
        raise ValueError("Colour temperature must be an integer")

    if not (0 <= colourtemp <= 1000):
        raise ValueError("Colour temperature must be between 0 and 1000")

    tasks = [
        control_device(device, "set_colourtemp", colourtemp)
        for device in devices
        if device.get('name') == device_name or all_devices
    ]

    if tasks:
        await asyncio.gather(*tasks)
        if all_devices:
            return [types.TextContent(type="text", text=f"All devices colour temperature set to {colourtemp}.")]
        return [types.TextContent(type="text", text=f"Device {device_name} colour temperature set to {colourtemp}.")]

    raise ValueError(f"Device {device_name} not found")

async def handle_set_mode(arguments):
    all_devices = arguments.get('all', False)
    device_name = arguments.get('device')
    mode = arguments.get('mode')

    if (not all_devices and not device_name) or not mode:
        raise ValueError("Device and mode must be provided")

    if mode not in ["white", "colour", "scene", "music"]:
        raise ValueError("Mode must be white, colour, scene, or music")

    tasks = [
        control_device(device, "set_mode", mode)
        for device in devices
        if device.get('name') == device_name or all_devices
    ]

    if tasks:
        await asyncio.gather(*tasks)
        if all_devices:
            return [types.TextContent(type="text", text=f"All devices mode set to {mode}.")]
        return [types.TextContent(type="text", text=f"Device {device_name} mode set to {mode}.")]

    raise ValueError(f"Device {device_name} not found")

async def handle_stop_music(arguments):
    device_name = arguments.get('device')
    all_devices = arguments.get('all', False)

    if not all_devices and not device_name:
        raise ValueError("Device or all must be provided")

    tasks = [
        stop_music(device.get('name'))
        for device in devices
        if device.get('name') == device_name or all_devices
    ]

    if tasks:
        await asyncio.gather(*tasks)
        if all_devices:
            return [types.TextContent(type="text", text=f"All devices stopped.")]
        return [types.TextContent(type="text", text=f"Device {device_name} stopped.")]

    raise ValueError(f"Device {device_name} not found")

async def stop_music(device_name):
    if device_name in music_processes:
        task = music_processes.pop(device_name)
        task.cancel()
        try:
            await task  # Ensure the task finishes
        except asyncio.CancelledError:
            pass  # Expected behavior
        return f"Music mode stopped for device {device_name}"
    
    return f"Music mode is not active for device {device_name}"

async def music_process(device, delay, freq_range):
    """Asynchronous task to process music mode for a device."""
    device_id = device['id']
    local_key = device['key']
    ip_address = device['ip']
    version = device["ver"]

    d = tinytuya.BulbDevice(device_id, ip_address, local_key)
    d.set_version(version)
    d.set_socketPersistent(True)

    try:
        while True:
            audio_data = audio_buffer.get()
            hue, saturation, value = parse_audio(audio_data, SAMPLE_RATE, freq_range)
            logging.info(f"Audio data processed: Hue: {hue}, Saturation: {saturation}, Value: {value}")
            d.set_hsv(hue / 360.0, saturation / 1000.0, value / 1000.0, nowait=True)
            await asyncio.sleep(delay)

    except asyncio.CancelledError:
        logging.info(f"Music process for {device['name']} cancelled.")
    except Exception as e:
        logging.error(f"Error in music_process: {e}")


def read_audio_thread():
    process = subprocess.Popen(
        [
            'pw-record',
            f'--target={SINKORSOURCE}',
            f'--rate={SAMPLE_RATE}',
            '-',
        ],
        env={'XDG_RUNTIME_DIR': XDG_RUNTIME_DIR},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    logging.info(f"Process parec listening to {SINKORSOURCE}. env XDG_RUNTIME_DIR={XDG_RUNTIME_DIR}")

    while True:
        data = process.stdout.read(SAMPLE_RATE)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(float)
        audio_buffer.put(audio_data)



async def handle_music(arguments):
    all_devices = arguments.get('all', False)
    device_name = arguments.get('device')
    delay = arguments.get('delay', 0.1)

    if not all_devices and not device_name:
        raise ValueError("device name must be provided")

    try:
        delay = float(delay)
    except ValueError:
        raise ValueError("delay must be a float")

    min_freq, max_freq = 100, 2000
    num_devices = len(devices)
    if num_devices == 0:
        raise ValueError("No devices configured")

    step = (max_freq - min_freq) // num_devices
    frequency_ranges = [(min_freq + i * step, min_freq + (i + 1) * step) for i in range(num_devices)]
    started_devices = []

    for i, device in enumerate(devices):
        if device.get('name') == device_name or all_devices:
            freq_range = frequency_ranges[i]
            task = asyncio.create_task(music_process(device, delay, freq_range))
            music_processes[device['name']] = task
            started_devices.append(f"{device['name']} ({freq_range[0]}-{freq_range[1]} Hz)")

    if started_devices:
        return [types.TextContent(type="text", text=f"Music mode started for {', '.join(started_devices)}")]

    raise ValueError("No matching devices found")


async def main():
    # Load devices from snapshot.json
    if load_devices().get("status") != "success":
        return

    # Start the audio reading thread
    audio_thread = threading.Thread(target=read_audio_thread, daemon=True)
    audio_thread.start()

    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        asyncio.create_task(server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="tuya_mcp_server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        ))

if __name__ == "__main__":
    asyncio.run(main())
