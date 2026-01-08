"""Internal utilities for generating embedded Viser visualizations."""

from __future__ import annotations

from pathlib import Path


def _get_client_html_path() -> Path:
    """Get path to the pre-built client HTML.

    Returns the path to client/build/index.html, checking both the installed
    package location and the development source tree.
    """
    # Check development path first (for editable installs and running from source).
    dev_path = Path(__file__).parent / "client" / "build" / "index.html"
    if dev_path.exists():
        return dev_path

    raise FileNotFoundError(
        "Client build not found. Please run 'npm run build' "
        "in the client directory first."
    )


def _get_client_html() -> str:
    """Load the pre-built client HTML."""
    return _get_client_html_path().read_text()
