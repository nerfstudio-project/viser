"""Generate typescript message definitions from Python dataclasses."""

import pathlib

from viser._typescript_interface_gen import generate_typescript_defs

if __name__ == "__main__":
    defs = generate_typescript_defs()
    target_path = pathlib.Path(__file__).parent / pathlib.Path(
        "client/src/WebsocketMessages.tsx"
    )
    assert target_path.exists()
    target_path.write_text(defs)
    print(f"Wrote to {target_path}")
