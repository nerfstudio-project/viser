from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, List, Tuple

if TYPE_CHECKING:
    from ._message_api import MessageApi


# TODO(by): we can add helpers for stuff like removing scene nodes, click events,
# etc here...

# @dataclasses.dataclass
# class _SceneHandleState:
#     name: str
#     api: MessageApi
#
#
# @dataclasses.dataclass(frozen=True)
# class SceneHandle:
#     _impl: _SceneHandleState
#


@dataclasses.dataclass
class _TransformControlsState:
    name: str
    api: MessageApi
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    last_updated: float

    update_cb: List[Callable[[TransformControlsHandle], None]]


@dataclasses.dataclass(frozen=True)
class TransformControlsState:
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    last_updated: float


@dataclasses.dataclass(frozen=True)
class TransformControlsHandle:
    _impl: _TransformControlsState

    def get_state(self) -> TransformControlsState:
        return TransformControlsState(
            self._impl.wxyz, self._impl.position, self._impl.last_updated
        )

    def on_update(
        self, func: Callable[[TransformControlsHandle], None]
    ) -> Callable[[TransformControlsHandle], None]:
        self._impl.update_cb.append(func)
        return func
