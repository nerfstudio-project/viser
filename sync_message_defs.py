"""Generate typescript message definitions from Python dataclasses."""

import pathlib
import subprocess

import viser.infra
from viser._messages import Message

if __name__ == "__main__":
    # Generate typescript source.
    defs = viser.infra.generate_typescript_interfaces(Message)

    # Write to file.
    target_path = pathlib.Path(__file__).parent / pathlib.Path(
        "src/viser/client/src/WebsocketMessages.ts"
    )
    assert target_path.exists()
    target_path.write_text(defs)
    print(f"Wrote to {target_path}")

    # Run prettier.
    subprocess.run(args=["npx", "prettier", "-w", str(target_path)], check=False)
