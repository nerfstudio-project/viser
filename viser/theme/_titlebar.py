from typing import Tuple, Literal, Optional, TypedDict


class TitlebarButton(TypedDict):
    text: Optional[str]
    icon: Optional[Literal["GitHub", "Description", "Keyboard"]]
    href: Optional[str]
    variant: Optional[Literal["text", "contained", "outlined"]]


class TitlebarImage(TypedDict):
    image_url: str
    image_alt: str
    href: Optional[str]


class TitlebarConfig(TypedDict):
    buttons: Optional[Tuple[TitlebarButton, ...]]
    image: Optional[TitlebarImage]
