# viser

**[ [API Reference](https://brentyi.github.io/viser) ]** &nbsp;&nbsp;&bull;&nbsp;&nbsp; `pip install viser`

![pyright](https://github.com/brentyi/viser/workflows/pyright/badge.svg)
![mypy](https://github.com/brentyi/viser/workflows/mypy/badge.svg)
![typescript](https://github.com/brentyi/viser/workflows/typescript-compile/badge.svg)
[![pypi](https://img.shields.io/pypi/pyversions/viser)](https://pypi.org/project/viser)

---

`viser` is a library for interactive 3D visualization + Python, inspired by
our favorite bits of the
[Nerfstudio viewer](https://github.com/nerfstudio-project/nerfstudio),
[Pangolin](https://github.com/stevenlovegrove/Pangolin),
[rviz](https://wiki.ros.org/rviz/), and
[meshcat](https://github.com/rdeits/meshcat).

Core features:

- Web interface for easy use on remote machines.
- Pure-Python API for sending 3D primitives to the browser.
- Python-configurable inputs: buttons, checkboxes, text inputs, sliders,
  dropdowns, gizmos.
- Support for multiple panels and view-synchronized connections.

## Running examples

```bash
# Clone the repository.
git clone https://github.com/brentyi/viser.git

# Install the package.
# You can also install via pip: `pip install viser`.
cd ./viser
pip install -e .

# Run an example.
pip install -r ./examples/requirements.txt
python ./examples/4_gui.py
```

After an example script is running, you can connect by navigating to the printed
URL (default: `http://localhost:8080`).

## Setup (client development)

This is only needed for client-side development. The automatically hosted viewer
should be sufficient otherwise.

```bash
cd ./viser/viser/client
yarn
yarn start
```

## Demos

### Interactive SMPL-X Example

https://user-images.githubusercontent.com/6992947/228734499-87d8a12a-df1a-4511-a4e0-0a46bd8532fd.mov


### Interactive NeRF rendering

(code not released)

https://user-images.githubusercontent.com/6992947/232163875-ff788455-f074-4bd3-9154-5330b5ed4733.mov
