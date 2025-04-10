"""Synchronize TypeScript definitions with Python.

This script:
1. Generates TypeScript message interfaces from Python dataclasses
2. Creates a version info file to keep client and server versions in sync
"""

import pathlib
import subprocess

import viser
import viser.infra
from viser._messages import Message

if __name__ == "__main__":
    # Generate TypeScript message interfaces
    defs = viser.infra.generate_typescript_interfaces(Message)

    # Write message interfaces to file
    target_path = pathlib.Path(__file__).parent / pathlib.Path(
        "src/viser/client/src/WebsocketMessages.ts"
    )
    # Create directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file even if it doesn't exist yet
    target_path.write_text(defs)
    print(f"{target_path} updated")

    # Generate version information file for client-server version compatibility checks
    version_path = pathlib.Path(__file__).parent / pathlib.Path(
        "src/viser/client/src/VersionInfo.ts"
    )
    # Create directory if it doesn't exist
    version_path.parent.mkdir(parents=True, exist_ok=True)

    version_content = f"""// Automatically generated file - do not edit manually.
// This is synchronized with the Python package version in viser/__init__.py.
export const VISER_VERSION = "{viser.__version__}";
"""
    version_path.write_text(version_content)
    print(f"Version info generated: {version_path}")

    # Run prettier on both files if it's available
    try:
        subprocess.run(
            args=["npx", "prettier", "-w", str(target_path), str(version_path)],
            check=False,
        )
    except FileNotFoundError:
        print("Warning: npx/prettier not found, skipping formatting")

    print("Synchronization complete")
