"""SparkJS Gaussian Splatting demo with SPZ files

Demonstrates using SparkJS for rendering Gaussian splats with spherical harmonics
support. Loads an SPZ file directly and displays it using
:meth:`~viser.SceneApi.add_gaussian_splats_sparkjs`. A red sphere is added to
verify proper depth compositing between splats and other 3D objects.

SPZ is a compressed format for Gaussian splats developed by Niantic Labs. You can
create SPZ files programmatically using the spz library, or download example files
from the SparkJS repository.
"""

import pathlib
import time
import tyro
import viser


def main(
    spz_path: pathlib.Path = pathlib.Path("test_splat.spz"),
) -> None:
    """Start a viser server with SparkJS Gaussian splats from an SPZ file.

    Args:
        spz_path: Path to an SPZ file containing Gaussian splat data.
    """
    # Check if SPZ file exists.
    if not spz_path.exists():
        print(f"Error: SPZ file not found at {spz_path}")
        print("\nPlease provide a valid path to an SPZ file:")
        print(f"  python {__file__} --spz-path <path/to/splat.spz>")
        print("\nYou can create a test SPZ file using the spz library:")
        print("  https://github.com/nianticlabs/spz")
        return

    # Load SPZ file as binary data.
    print(f"Loading SPZ file: {spz_path}")
    with open(spz_path, "rb") as f:
        spz_binary = f.read()

    file_size_mb = len(spz_binary) / (1024 * 1024)
    print(f"Loaded SPZ file: {file_size_mb:.2f} MB")

    # Start viser server.
    server = viser.ViserServer()
    print("Server started! Visit the URL above to view the splats.")

    # Add Gaussian splats using SparkJS.
    server.scene.add_gaussian_splats_sparkjs(
        name="/splats",
        spz_binary=spz_binary,
    )
    print("Added Gaussian splats to scene")

    # Add a red sphere at the origin for depth compositing verification.
    # If depth compositing works correctly, splats closer to the camera should
    # occlude the sphere, and the sphere should occlude more distant splats.
    server.scene.add_icosphere(
        name="/depth_test_sphere",
        radius=0.1,
        color=(255, 0, 0),
        position=(0.0, 0.0, 0.0),
    )
    print("Added red sphere for depth compositing test")

    # Keep the server running.
    print("\nPress Ctrl+C to exit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    tyro.cli(main)
