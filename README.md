# viser

![pointcloud_preview](./viser.png)

## TODO

Scene tree

- [x] useState prototype
- [x] useState -> zustand

Websocket connection

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
  - [ ] Camera control
  - [x] Ensure message ordering

- Serialization

  - [x] JSON
  - [x] JSON -> msgpack
  - [x] Automate synchronization of typescript / dataclass definitions

- UI

  - [ ] Remove visibility toggle interface
  - [ ] Regex filter for visibility
  - [ ] Icons for scene node type

- Exporting
  - [ ] Video export
  - [ ] SVG export
