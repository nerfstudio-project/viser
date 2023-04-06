"""Generate typescript message definitions from Python dataclasses."""

import pathlib
import subprocess

from viser._typescript_interface_gen import generate_typescript_defs

if __name__ == "__main__":
    # Generate typescript source.
    defs = generate_typescript_defs()

    # Write to file.
    target_path = pathlib.Path(__file__).parent / pathlib.Path(
        "viser/client/src/WebsocketMessages.tsx"
    )
    assert target_path.exists()
    target_path.write_text(defs)
    print(f"Wrote to {target_path}")

    # Run prettier.
    subprocess.run(args=["prettier", "-w", str(target_path)])
