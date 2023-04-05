# viser

`viser` is a library for web-based 3D visualization + Python, inspired by the
best bits of the
[Nerfstudio viewer](https://github.com/nerfstudio-project/nerfstudio),
[Pangolin](https://github.com/stevenlovegrove/Pangolin),
[rviz](https://wiki.ros.org/rviz/), and
[meshcat](https://github.com/rdeits/meshcat).

Core features:

- A websocket interface for easy use on remote machines.
- Visualization primitives: coordinate axes, frustums, point clouds, meshes,
  images.
- Python-configurable inputs: buttons, checkboxes, text inputs, sliders,
  dropdowns, gizmos.
- Support for multiple panels and view-synchronized connections; helpful for
  side-by-side comparisons.

---

## Setup

```bash
# Clone the repository.
git clone https://github.com/brentyi/viser.git

# Install the package.
cd ./viser
pip install -e .

# Run an example.
python ./examples/4_gui.py
```

After an example script is running, you can connect by navigating to the printed
URL (default: `http://localhost:8080`).

## Setup (client development)

This is only needed for client-side development. The automatically hosted viewer
should be sufficient otherwise.

```bash
cd ./viser/viser/client
npm install
npm start
```

---

https://user-images.githubusercontent.com/6992947/228734499-87d8a12a-df1a-4511-a4e0-0a46bd8532fd.mov

## TODO

Python-controllable GUI

- [x] Plumbing (new broadcast/client interfaces etc)
- [x] Primitives
  - [x] Select / drop-down
  - [x] Checkbox
  - [x] Slider
  - [x] Basic textbox
  - [x] 2D vector
  - [x] 3D vector
  - [x] Button
- [x] Commands
  - [x] .value(), last_updated()
  - [x] Callback interface
  - [x] Set value from Python
  - [x] Allow disabling
  - [x] Remove GUI element
- [x] Synchronize GUIs across clients (for broadcasted)
- [x] Folders

Scene tree

- [x] useState prototype
- [x] useState -> zustand

- Websocket connection

  - [x] Stateful server
  - [x] Redundant message culling
  - [x] Basic handling of conflicting ports
  - [x] Automatic HTTP server for front-end

- Camera controls

  - [x] Orbit controls
  - [ ] Keyboard

- Message types

  - [x] Coordinate frame
  - [x] Point cloud
    - [x] Naive serialization
    - [x] Directly access `.data`
  - [x] Camera frustum
  - [x] Image
  - [x] Video stream (seems fine to just images for this)
  - [x] Background set
  - [x] Camera read
  - [ ] Camera write
  - [x] Ensure message ordering
  - [x] Meshes!!
  - [x] Transform controls
  - [x] Environment map
  - [x] Set visibility

- Serialization

  - [x] JSON
  - [x] JSON -> msgpack
  - [x] Automate synchronization of typescript / dataclass definitions

- UI

  - [ ] Icons for scene node type

- Exporting

  - [x] Background download
