from typing import get_type_hints

from viser.infra._messages import Message


def test_get_annotations() -> None:
    """Check that we can read the type annotations from all messages.

    This is to guard against use of Python annotations that can't be inspected at runtime.
    We could also use `eval_type_backport`: https://github.com/alexmojaki/eval_type_backport
    """
    for cls in Message.get_subclasses():
        try:
            get_type_hints(cls)
        except TypeError as e:
            raise AssertionError(f"Error reading type hints for {cls}") from e
