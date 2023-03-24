# viser


## Setup


#### Client
```bash
cd viser/client
npm install
npm start
```

#### Server

```bash
cd viser/
pip install -e .
python ./examples/4_gui.py  # Or other example
```

---

![pointcloud_preview](./viser.png)

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
  - [ ] Multiple "servers"? Conflicting ports?

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
  - [ ] Lights?
  - [x] Set visibility

- Serialization

  - [x] JSON
  - [x] JSON -> msgpack
  - [x] Automate synchronization of typescript / dataclass definitions

- UI

  - [ ] Icons for scene node type
  - [ ] 

- Exporting
  - [x] Background download
  - [ ] Video export
  - [ ] SVG export
