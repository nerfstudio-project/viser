import json
import struct

import numpy as np
import trimesh
from trimesh.visual import ColorVisuals

import viser
from viser._scene_api import _trimesh_with_transparency


def _glb_materials(glb_data: bytes):
    """Parse the JSON chunk of a GLB payload, return its materials (if any)."""
    json_length = struct.unpack("<I", glb_data[12:16])[0]
    gltf = json.loads(glb_data[20 : 20 + json_length])
    return gltf.get("materials")


def _queued_glb_data(server: viser.ViserServer, name: str) -> bytes:
    """Return the GLB payload of the most recent queued add-message for ``name``."""
    found = None
    for message in server._websock_server._broadcast_buffer.message_from_id.values():
        if getattr(message, "name", None) == name and hasattr(message, "props"):
            found = message.props
    assert found is not None, f"no queued message for {name}"
    return found.glb_data


def test_trimesh_mesh_helpers() -> None:
    server = viser.ViserServer(port=0, verbose=False)
    try:
        mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        server.scene.add_mesh_trimesh("/mesh", mesh)
        server.scene.add_batched_meshes_trimesh(
            "/batched",
            mesh,
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            batched_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        )
    finally:
        server.stop()


def test_add_mesh_trimesh_preserves_face_color_transparency() -> None:
    """Repro for https://github.com/viser-project/viser/issues/525.

    RGBA face colors must survive the trimesh -> GLB export as a transparent
    material instead of being rendered fully opaque.
    """
    server = viser.ViserServer(port=0, verbose=False)
    try:
        mesh = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=16)
        mesh.visual.face_colors = [255, 0, 0, 76]
        server.scene.add_mesh_trimesh("/transparent", mesh)

        materials = _glb_materials(_queued_glb_data(server, "/transparent"))
        assert materials is not None, "GLB export is missing materials"
        assert materials[0].get("alphaMode") == "BLEND"

        # The caller's mesh must not be mutated.
        assert isinstance(mesh.visual, ColorVisuals)
        assert (np.asanyarray(mesh.visual.face_colors)[:, 3] == 76).all()
    finally:
        server.stop()


def test_add_mesh_trimesh_opaque_mesh_unchanged() -> None:
    """Fully opaque meshes keep the legacy (material-free) export path."""
    server = viser.ViserServer(port=0, verbose=False)
    try:
        mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        mesh.visual.face_colors = [0, 0, 255, 255]
        server.scene.add_mesh_trimesh("/opaque", mesh)

        assert _glb_materials(_queued_glb_data(server, "/opaque")) is None
        assert isinstance(mesh.visual, ColorVisuals)
    finally:
        server.stop()


def test_trimesh_with_transparency_vertex_colors() -> None:
    """Vertex-color transparency is fixed; opaque input returns as-is."""
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.visual.vertex_colors = np.tile(
        np.array([[0, 255, 0, 153]], dtype=np.uint8), (len(mesh.vertices), 1)
    )
    converted = _trimesh_with_transparency(mesh)
    assert converted is not mesh
    assert isinstance(mesh.visual, ColorVisuals)

    opaque = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    assert _trimesh_with_transparency(opaque) is opaque
