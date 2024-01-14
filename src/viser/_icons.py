import base64
import tarfile
from pathlib import Path

from ._icons_enum import IconName

ICONS_DIR = Path(__file__).absolute().parent / "_icons"


def base64_from_icon(icon_name: IconName) -> str:
    """Read an icon and encode it via base64."""
    assert isinstance(icon_name, str)
    icons_tarball = ICONS_DIR / "tabler-icons.tar"

    with tarfile.open(icons_tarball) as tar:
        icon_file = tar.extractfile(f"{icon_name}.svg")
        assert icon_file is not None
        out = icon_file.read()

    return base64.b64encode(out).decode("ascii")
