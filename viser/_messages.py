"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import base64
import dataclasses
import functools
import io
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import imageio.v3 as iio
import msgpack
import numpy as onp
import numpy.typing as onpt
from typing_extensions import Literal, assert_never


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
        """Convert a Python Message object into bytes."""
        mapping = {k: _prepare_for_serialization(v) for k, v in vars(self).items()}
        out = msgpack.packb({"type": self.type, **mapping})
        assert isinstance(out, bytes)
        return out

    @staticmethod
    def deserialize(message: bytes) -> Message:
        """Convert bytes into a Python Message object."""
        mapping = msgpack.unpackb(message)
        message_type = Message._subclass_from_type_string()[mapping.pop("type")]
        return message_type(**mapping)

    @staticmethod
    @functools.lru_cache
    def _subclass_from_type_string() -> Dict[str, Type[Message]]:
        subclasses = Message.get_subclasses()
        return {s.type: s for s in subclasses}

    @staticmethod
    def get_subclasses() -> List[Type[Message]]:
        """Recursively get message subclasses."""

        def _get_subclasses(typ: Type[Message]) -> List[Type[Message]]:
            out = []
            for sub in typ.__subclasses__():
                out.append(sub)
                out.extend(_get_subclasses(sub))
            return out

        return _get_subclasses(Message)


@dataclasses.dataclass
class ViewerCameraMessage(Message):
    """Message for a posed viewer camera.
    Pose is in the form T_world_camera, OpenCV convention, +Z forward."""

    type: ClassVar[str] = "viewer_camera"
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    fov: float
    aspect: float
    # Should we include near and far?


@dataclasses.dataclass
class CameraFrustumMessage(Message):
    """Variant of CameraMessage used for visualizing camera frustums.

    OpenCV convention, +Z forward."""

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
class BackgroundImageMessage(Message):
    """Message for rendering a background image."""

    type: ClassVar[str] = "background_image"
    media_type: Literal["image/jpeg", "image/png"]
    base64_data: str

    @staticmethod
    def encode(
        image: onpt.NDArray[onp.uint8],
        format: Literal["png", "jpeg"] = "jpeg",
        quality: Optional[int] = None,
    ) -> BackgroundImageMessage:
        with io.BytesIO() as data_buffer:
            if format == "png":
                media_type = "image/png"
                iio.imwrite(data_buffer, image, format="PNG")
            elif format == "jpeg":
                media_type = "image/jpeg"
                iio.imwrite(
                    data_buffer,
                    image,
                    format="JPEG",
                    quality=75 if quality is None else quality,
                )
            else:
                assert_never(format)

            base64_data = base64.b64encode(data_buffer.getvalue()).decode("ascii")

        return BackgroundImageMessage(media_type=media_type, base64_data=base64_data)


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
    def encode(
        name: str,
        image: onpt.NDArray[onp.uint8],
        render_width: float,
        render_height: float,
        format: Literal["png", "jpeg"] = "jpeg",
        quality: Optional[int] = None,
    ) -> ImageMessage:
        proxy = BackgroundImageMessage.encode(image, format=format, quality=quality)
        return ImageMessage(
            name=name,
            media_type=proxy.media_type,
            base64_data=proxy.base64_data,
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
