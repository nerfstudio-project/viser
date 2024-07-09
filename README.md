<h1 align="left">
    <img alt="viser logo" src="https://viser.studio/latest/_static/logo.svg" width="auto" height="30" />
    viser
    <img alt="viser logo" src="https://viser.studio/latest/_static/logo.svg" width="auto" height="30" />
</h1>

<p align="left">
    <img alt="pyright" src="https://github.com/nerfstudio-project/viser/workflows/pyright/badge.svg?branch=main" />
    <img alt="typescript-compile" src="https://github.com/nerfstudio-project/viser/workflows/typescript-compile/badge.svg?branch=main" />
    <a href="https://pypi.org/project/viser/">
        <img alt="codecov" src="https://img.shields.io/pypi/pyversions/viser" />
    </a>
</p>

`viser` is a library for interactive 3D visualization in Python.

Features include:

- API for visualizing 3D primitives
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos)
- Programmatic camera control and rendering
- An entirely web-based client, for easy use over SSH!

For usage and API reference, see our <a href="https://viser.studio/latest">documentation</a>.

## Installation

You can install `viser` with `pip`:

```bash
pip install viser
```

To include example dependencies:

```bash
pip install viser[examples]
```

After an example script is running, you can connect by navigating to the printed
URL (default: `http://localhost:8080`).

See also: our [development docs](https://viser.studio/latest/development/).

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

## Acknowledgements

`viser` is heavily inspired by packages like
[Pangolin](https://github.com/stevenlovegrove/Pangolin),
[rviz](https://wiki.ros.org/rviz/),
[meshcat](https://github.com/rdeits/meshcat), and
[Gradio](https://github.com/gradio-app/gradio).
It's made possible by several open-source projects.

The web client is implemented using [React](https://react.dev/), with:

- [Vite](https://vitejs.dev/) / [Rollup](https://rollupjs.org/) for bundling
- [three.js](https://threejs.org/) via [react-three-fiber](https://github.com/pmndrs/react-three-fiber) and [drei](https://github.com/pmndrs/drei)
- [Mantine](https://mantine.dev/) for UI components
- [zustand](https://github.com/pmndrs/zustand) for state management
- [vanilla-extract](https://vanilla-extract.style/) for stylesheets

The Python API communicates via [msgpack](https://msgpack.org/index.html) and [websockets](https://websockets.readthedocs.io/en/stable/index.html).
