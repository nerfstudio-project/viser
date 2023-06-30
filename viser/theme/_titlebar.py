from typing import Tuple, Literal, Optional, TypedDict


class TitlebarButton(TypedDict):
    """A link-only button that appears in the Titlebar."""

    text: Optional[str]
    icon: Optional[Literal["GitHub", "Description", "Keyboard"]]
    href: Optional[str]
    variant: Optional[Literal["text", "contained", "outlined"]]


class TitlebarImage(TypedDict):
    """An image that appears on the titlebar."""

    image_url_light: str
    image_url_dark: Optional[str]
    image_alt: str
    href: Optional[str]


class TitlebarConfig(TypedDict):
    """Configure the content that appears in the titlebar."""

    buttons: Optional[Tuple[TitlebarButton, ...]]
    image: Optional[TitlebarImage]
