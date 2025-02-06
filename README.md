# Tuya MCP Client CLI

[![smithery badge](https://smithery.ai/badge/@cabra-lat/tuyactl)](https://smithery.ai/server/@cabra-lat/tuyactl)

This project provides a command-line interface (`tuyactl`) for controlling Tuya devices. It interacts with a separate Tuya MCP server (not covered in this guide).

## Requirements

*   **uv:** A fast and modern Python package installer and runner.  Install it by following the instructions on the [uv documentation site](https://docs.astral.sh/uv/installation/).
*   **Tuya Local Keys:** You will need the local keys for your Tuya devices. Follow the [tinytuya setup wizard](https://github.com/jasonacox/tinytuya#setup-wizard---getting-local-keys) to obtain these.

## Quick Start

### Installing via Smithery

To install Tuya MCP Client CLI for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@cabra-lat/tuyactl):

```bash
npx -y @smithery/cli install @cabra-lat/tuyactl --client claude
```

### Run the Server

1.  **Install `uv`:**

    Follow the official installation instructions on the [uv documentation site](https://docs.astral.sh/uv/installation/). The recommended method is to use the standalone installer, which you can download and run with the following command:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Obtain Tuya Local Keys:**

    Follow the [tinytuya setup wizard](https://github.com/jasonacox/tinytuya#setup-wizard---getting-local-keys) to get the local keys for your Tuya devices.  Place the resulting `snapshot.json` file in your home directory (`~`). You can customize the location of this file using environment variables (see below).

3.  **Run `tuyactl`:**

    To see the available commands and options, run:

    ```bash
    tuyactl --help
    ```

    To execute a specific command, use the following syntax:

    ```bash
    tuyactl <command> [options]
    ```

    Replace `<command>` with one of the available commands: `list`, `on`, `off`, `color`, `brightness`, `temperature`, `mode`, `music`.  Use the `--
help` option to see the available options for each command.

    For example, to list all your Tuya devices, run:

    ```bash
    tuyactl list
    ```

## Configuration

*   **`snapshot.json` Location:** You can customize the location of the `snapshot.json` file (containing your Tuya device keys) using environment va
riables. (Details on this to be added later).
