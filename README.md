<h1 align="left">
    <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="30" height="auto" />
    Viser
    <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="30" height="auto" />
</h1>

<p align="left">
    <img alt="pyright" src="https://github.com/nerfstudio-project/viser/actions/workflows/pyright.yml/badge.svg" />
    <img alt="typescript-compile" src="https://github.com/nerfstudio-project/viser/actions/workflows/typescript-compile.yml/badge.svg" />
    <a href="https://pypi.org/project/viser/">
        <img alt="codecov" src="https://img.shields.io/pypi/pyversions/viser" />
    </a>
</p>

Viser is a 3D visualization library for computer vision and robotics in Python.

Features include:

- API for visualizing 3D primitives.
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos).
- Programmatic camera control and rendering.
- An entirely web-based client, for easy use over SSH!

The goal is to provide primitives that are (1) easy for simple visualization tasks, but (2) can be composed into more elaborate interfaces. For more about design goals, see the [technical report](https://arxiv.org/abs/2507.22885).

Examples and documentation: https://viser.studio

## Installation

You can install `viser` with `pip`:

```bash
pip install viser            # Core dependencies only.
pip install viser[examples]  # To include example dependencies.
```

That's it! To learn more, we recommend looking at the examples in the [documentation](https://viser.studio/).

## Citation

To cite Viser in your work, you can use the BibTeX for our [technical report](https://arxiv.org/abs/2507.22885):

```
@misc{yi2025viser,
      title={Viser: Imperative, Web-based 3D Visualization in Python},
      author={Brent Yi and Chung Min Kim and Justin Kerr and Gina Wu and Rebecca Feng and Anthony Zhang and Jonas Kulhanek and Hongsuk Choi and Yi Ma and Matthew Tancik and Angjoo Kanazawa},
      year={2025},
      eprint={2507.22885},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.22885},
}
```

## Acknowledgements

`viser` is heavily inspired by packages like
[Pangolin](https://github.com/stevenlovegrove/Pangolin),
[Dear ImGui](https://github.com/ocornut/imgui),
[rviz](https://wiki.ros.org/rviz/),
[meshcat](https://github.com/rdeits/meshcat), and
[Gradio](https://github.com/gradio-app/gradio).

The web client is implemented using [React](https://react.dev/), with:

- [Vite](https://vitejs.dev/) / [Rollup](https://rollupjs.org/) for bundling
- [three.js](https://threejs.org/) via [react-three-fiber](https://github.com/pmndrs/react-three-fiber) and [drei](https://github.com/pmndrs/drei)
- [Mantine](https://mantine.dev/) for UI components
- [zustand](https://github.com/pmndrs/zustand) for state management
- [vanilla-extract](https://vanilla-extract.style/) for stylesheets

Thanks to the authors of these projects for open-sourcing their work!
