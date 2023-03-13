"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import base64
import dataclasses
import io
from typing import Any, ClassVar, Tuple

import msgpack
import numpy as onp
import numpy.typing as onpt
from PIL import Image
from typing_extensions import Literal


def _prepare_for_serialization(value: Any) -> Any:
    """Prepare any special types for serialization. Currently just maps numpy arrays to
    their underlying data buffers."""

    if isinstance(value, onp.ndarray):
        return value.data if value.data.c_contiguous else value.copy().data
    else:
        return value


class Message:
    """Base message type for controlling our viewer."""

    type: ClassVar[str]

    def serialize(self) -> bytes:
        mapping = {k: _prepare_for_serialization(v) for k, v in vars(self).items()}
        out = msgpack.packb({"type": self.type, **mapping})
        assert isinstance(out, bytes)
        return out


@dataclasses.dataclass
class ViewerCameraMessage(Message):
    """Message for a posed viewer camera."""

    type: ClassVar[str] = "viewer_camera"
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    fov: float
    aspect: float
    # Should we include near and far?


@dataclasses.dataclass
class CameraFrustumMessage(Message):
    """Variant of CameraMessage used for visualizing camera frustums."""

    type: ClassVar[str] = "camera_frustum"
    name: str
    fov: float
    aspect: float
    scale: float = 0.3


@dataclasses.dataclass
class FrameMessage(Message):
    """Coordinate frame message.

    Position and orientation should follow a `T_parent_local` convention, which
    corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`."""

    type: ClassVar[str] = "frame"
    name: str
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    show_axes: bool = True
    scale: float = 0.5


@dataclasses.dataclass
class PointCloudMessage(Message):
    """Message for rendering"""

    type: ClassVar[str] = "point_cloud"
    name: str
    position_f32: onpt.NDArray[onp.float32]
    color_uint8: onpt.NDArray[onp.uint8]
    point_size: float = 0.1

    def __post_init__(self):
        assert self.position_f32.dtype == onp.float32
        assert self.color_uint8.dtype == onp.uint8
        assert self.position_f32.shape == self.color_uint8.shape
        assert self.position_f32.shape[-1] == 3


@dataclasses.dataclass
class ImageMessage(Message):
    """Message for rendering 2D images."""

    # Note: it might be faster to do the bytes->base64 conversion on the client.
    # Potentially worth revisiting.

    type: ClassVar[str] = "image"
    name: str
    media_type: Literal["image/jpeg", "image/png"]
    base64_data: str
    render_width: float
    render_height: float

    @staticmethod
    def from_array(
        name: str,
        array: onpt.NDArray[onp.uint8],
        render_width: float,
        render_height: float,
    ) -> ImageMessage:
        return ImageMessage.from_image(
            name=name,
            image=Image.fromarray(array),
            render_width=render_width,
            render_height=render_height,
        )

    @staticmethod
    def from_image(
        name: str,
        image: Image.Image,
        render_width: float,
        render_height: float,
    ) -> ImageMessage:
        with io.BytesIO() as data_buffer:
            # Use a PNG when an alpha channel is required, otherwise JPEG.
            # In the future, we could expose more granular controls for this.
            if image.mode == "RGBA":
                media_type = "image/png"
                image.save(data_buffer, format="PNG")
            elif image.mode == "RGB":
                media_type = "image/jpeg"
                image.save(data_buffer, format="JPEG", quality=60)
            else:
                assert False, f"Unexpected image mode {image.mode}"

            return ImageMessage(
                name=name,
                media_type="image/png",
                base64_data=base64.b64encode(data_buffer.getvalue()).decode("ascii"),
                render_width=render_width,
                render_height=render_height,
            )


@dataclasses.dataclass
class RemoveSceneNodeMessage(Message):
    """Remove a particular node from the scene."""

    type: ClassVar[str] = "remove_scene_node"
    name: str


@dataclasses.dataclass
class ResetSceneMessage(Message):
    """Reset scene."""

    type: ClassVar[str] = "reset_scene"
