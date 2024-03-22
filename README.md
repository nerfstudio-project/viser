<p align="center">
    <!--
    Note that this README will be used for both GitHub and PyPI.
    We therefore:
    - Keep all image URLs absolute.
    - In the GitHub action we use for publishing, strip some HTML tags that aren't supported by PyPI.
    -->
    <!-- pypi-strip -->
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://viser.studio/_static/viser_banner_dark.svg" />
    <!-- /pypi-strip -->
        <img alt="tyro logo" src="https://viser.studio/_static/viser_banner.svg" />
    <!-- pypi-strip -->
    </picture>
    <!-- /pypi-strip -->

</p>

<p align="center">
    <strong><code>pip install viser</code></strong>
    &nbsp;&nbsp;&bull;&nbsp;&nbsp;
    <strong><a href="https://viser.studio">Documentation</a></strong>
</p>

<p align="center">
    <img alt="pyright" src="https://github.com/nerfstudio-project/viser/workflows/pyright/badge.svg?branch=main" />
    <img alt="mypy" src="https://github.com/nerfstudio-project/viser/workflows/mypy/badge.svg?branch=main" />
    <img alt="typescript-compile" src="https://github.com/nerfstudio-project/viser/workflows/typescript-compile/badge.svg?branch=main" />
    <a href="https://pypi.org/project/viser/">
        <img alt="codecov" src="https://img.shields.io/pypi/pyversions/viser" />
    </a>
</p>

**`viser`** is a library for interactive 3D visualization + Python, inspired by
tools like [Pangolin](https://github.com/stevenlovegrove/Pangolin),
[rviz](https://wiki.ros.org/rviz/),
[meshcat](https://github.com/rdeits/meshcat), and
[Gradio](https://github.com/gradio-app/gradio). It's designed to support
applications in 3D vision and robotics.

As a standalone visualization tool, `viser` features include:

- Python API for visualizing 3D primitives in a web browser.
- Python-configurable GUI elements: buttons, checkboxes, text inputs, sliders,
  dropdowns, and more.
- A [meshcat](https://github.com/rdeits/meshcat) and
  [tf](http://wiki.ros.org/tf2)-inspired coordinate frame tree.

The `viser.infra` backend can also be used to build custom web applications. It
supports:

- Websocket / HTTP server management, on a shared port.
- Asynchronous server/client communication infrastructure.
- Client state persistence logic.
- Typed serialization; synchronization between Python dataclass and TypeScript
  interfaces.

## Installation

You can install `viser` with `pip`:

```bash
pip install viser
```

To run examples:

```bash
# Clone the repository.
git clone https://github.com/nerfstudio-project/viser.git

# Install the package.
# You can also install via pip: `pip install viser`.
cd ./viser
pip install -e .

# Run an example.
pip install -e .[examples]
python ./examples/02_gui.py
```

After an example script is running, you can connect by navigating to the printed
URL (default: `http://localhost:8080`).

See also: our [development docs](https://viser.studio/development/).

## Examples

**Point cloud visualization**

https://github.com/nerfstudio-project/viser/assets/6992947/df35c6ee-78a3-43ad-a2c7-1dddf83f7458

Source: `./examples/07_record3d_visualizer.py`

**Gaussian splatting visualization**

https://github.com/nerfstudio-project/viser/assets/6992947/c51b4871-6cc8-4987-8751-2bf186bcb1ae

Source:
[WangFeng18/3d-gaussian-splatting](https://github.com/WangFeng18/3d-gaussian-splatting)
and
[heheyas/gaussian_splatting_3d](https://github.com/heheyas/gaussian_splatting_3d).

**SMPLX visualizer**

https://github.com/nerfstudio-project/viser/assets/6992947/78ba0e09-612d-4678-abf3-beaeeffddb01

Source: `./example/08_smpl_visualizer.py`
