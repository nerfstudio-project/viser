import base64
from pathlib import Path

from ._icons_enum import Icon

ICONS_DIR = Path(__file__).absolute().parent / "_icons"


def base64_from_icon(icon: Icon) -> str:
    """Read an icon and encode it via base64."""
    icon_name = icon.value
    assert isinstance(icon_name, str)
    icon_file = ICONS_DIR / f"{icon_name}.svg"
    assert icon_file.exists(), f"Icon {icon_name} does not exist!"
    return base64.b64encode(icon_file.read_bytes()).decode("ascii")
