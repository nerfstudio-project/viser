class Button:
    icon: str
    label: str

    def __init__(self, name: str, icon: str, callback: Callable[[], None]) -> None: ...

class GuiApiMixin:
    def add_gui_button(
        self, name: str, icon: str, callback: Callable[[], None]
    ) -> Button:
        """
        Add gui button
        """
        ...
