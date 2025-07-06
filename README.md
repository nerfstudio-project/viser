<h1 align="left">
    <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="30" height="auto" />
    viser
    <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="30" height="auto" />
</h1>

<p align="left">
    <img alt="pyright" src="https://github.com/nerfstudio-project/viser/actions/workflows/pyright.yml/badge.svg" />
    <img alt="typescript-compile" src="https://github.com/nerfstudio-project/viser/actions/workflows/typescript-compile.yml/badge.svg" />
    <a href="https://pypi.org/project/viser/">
        <img alt="codecov" src="https://img.shields.io/pypi/pyversions/viser" />
    </a>
</p>

**viser** is a library for interactive 3D visualization in Python.

Features include:

- API for visualizing 3D primitives
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos)
- Programmatic camera control and rendering
- An entirely web-based client, for easy use over SSH!

Docs and examples: https://viser.studio.

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

See also: our [development docs](https://viser.studio/main/development/).

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
