import tarfile
from pathlib import Path

from ._icons_enum import IconName

ICONS_DIR = Path(__file__).absolute().parent / "_icons"


def svg_from_icon(icon_name: IconName) -> str:
    """Read an icon and return it as a UTF string; we expect this to be an
    <svg /> tag."""
    assert isinstance(icon_name, str)
    icons_tarball = ICONS_DIR / "tabler-icons.tar"

    with tarfile.open(icons_tarball) as tar:
        icon_file = tar.extractfile(f"{icon_name}.svg")
        assert icon_file is not None
        out = icon_file.read()

    return out.decode("utf-8")
