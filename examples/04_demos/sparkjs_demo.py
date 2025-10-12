"""SparkJS Gaussian Splatting demo

Demonstrates using SparkJS for rendering Gaussian splats with spherical harmonics
support. Loads a PLY file, converts it to SPZ format, and displays it using
:meth:`~viser.SceneApi.add_gaussian_splats_sparkjs`. A red sphere is added to
verify proper depth compositing between splats and other 3D objects.

To run this demo, you'll need to install the SPZ library:
    git clone https://github.com/nianticlabs/spz.git
    cd spz
    pip install .
"""

import pathlib
import time
import tyro
import viser


def main(
    ply_path: pathlib.Path = pathlib.Path("ale_splat.ply"),
) -> None:
    """Start a viser server with SparkJS Gaussian splats.

    Args:
        ply_path: Path to a PLY file containing Gaussian splat data.
    """
    # Check if SPZ library is installed.
    try:
        import spz  # type: ignore
    except ImportError:
        print("Error: SPZ library not found.")
        print("\nTo use this demo, please install the SPZ library:")
        print("  git clone https://github.com/nianticlabs/spz.git")
        print("  cd spz")
        print("  pip install .")
        return

    # Check if PLY file exists.
    if not ply_path.exists():
        print(f"Error: PLY file not found at {ply_path}")
        print("\nPlease provide a valid path to a Gaussian splat PLY file:")
        print(f"  python {__file__} --ply-path <path/to/splat.ply>")
        return

    # Convert PLY to SPZ format.
    print(f"Loading PLY file: {ply_path}")
    unpack_options = spz.UnpackOptions()  # type: ignore
    # Don't convert coordinate systems - keep the original PLY coordinate system.
    # SPZ will preserve the data as-is, and Viser will display it correctly.
    # unpack_options.to_coord = spz.CoordinateSystem.RDF  # type: ignore

    # Try loading the PLY file. If it loads with 0 points due to comment lines
    # (common with Nerfstudio exports), automatically fix the PLY by removing
    # comment lines and retry.
    cloud = spz.load_splat_from_ply(str(ply_path), unpack_options)  # type: ignore

    # Check if the cloud loaded correctly. If it has 0 points, it likely failed
    # due to comment lines in the header.
    if cloud.num_points == 0:
        print(
            "\nWarning: PLY loaded with 0 points. This is likely a Nerfstudio-exported PLY file."
        )
        print("Attempting to fix by removing comment lines from PLY header...\n")

        # Fix the PLY file by removing comment lines.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_fixed:
            fixed_ply_path = pathlib.Path(tmp_fixed.name)

        # Read and fix the PLY file.
        with open(ply_path, "rb") as f:
            lines = []
            # Read header line by line until we hit binary data.
            while True:
                line = f.readline()
                if not line:
                    break

                # Decode the line.
                line_str = line.decode("utf-8", errors="ignore").strip()

                # Skip comment lines.
                if line_str.startswith("comment"):
                    continue

                # Add non-comment lines.
                lines.append(line)

                # If we hit end_header, read the rest as binary.
                if line_str == "end_header":
                    binary_data = f.read()
                    break

        # Write the fixed PLY.
        with open(fixed_ply_path, "wb") as f:
            for line in lines:
                f.write(line)
            f.write(binary_data)

        print(f"Fixed PLY written to temporary file: {fixed_ply_path}")

        # Try loading the fixed PLY.
        try:
            cloud = spz.load_splat_from_ply(str(fixed_ply_path), unpack_options)  # type: ignore
            print("Successfully loaded fixed PLY file!")
        finally:
            # Clean up the temporary fixed PLY file.
            fixed_ply_path.unlink()

    print(f"Loaded {cloud.num_points} Gaussians from PLY")

    # Save as compressed SPZ format in memory.
    pack_options = spz.PackOptions()  # type: ignore
    # No coordinate system conversion needed.
    # pack_options.from_coord = spz.CoordinateSystem.RDF  # type: ignore

    # Save to temporary file and read back as bytes.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".spz", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    spz.save_spz(cloud, pack_options, tmp_path)  # type: ignore
    with open(tmp_path, "rb") as f:
        spz_binary = f.read()

    # Clean up temporary file.
    pathlib.Path(tmp_path).unlink()

    # Calculate compression ratio.
    import os

    ply_size = os.path.getsize(ply_path)
    spz_size = len(spz_binary)
    compression_ratio = ply_size / spz_size
    print(
        f"Compressed from {ply_size / 1024 / 1024:.1f}MB to {spz_size / 1024 / 1024:.1f}MB"
    )
    print(f"Compression ratio: {compression_ratio:.1f}x smaller")

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
