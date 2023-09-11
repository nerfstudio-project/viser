<h1>
  viser
</h1>

**`pip install viser`** &nbsp;&nbsp;&bull;&nbsp;&nbsp; **[
[API Reference](https://nerfstudio-project.github.io/viser) ]**

![pyright](https://github.com/nerfstudio-project/viser/workflows/pyright/badge.svg)
![mypy](https://github.com/nerfstudio-project/viser/workflows/mypy/badge.svg)
![typescript](https://github.com/nerfstudio-project/viser/workflows/typescript-compile/badge.svg)
[![pypi](https://img.shields.io/pypi/pyversions/viser)](https://pypi.org/project/viser)

---

`viser` is a library for interactive 3D visualization + Python, inspired by
tools like [Pangolin](https://github.com/stevenlovegrove/Pangolin),
[rviz](https://wiki.ros.org/rviz/),
[meshcat](https://github.com/rdeits/meshcat), and
[Gradio](https://github.com/gradio-app/gradio).

As a standalone visualization tool, `viser` features include:

- Web interface for easy use on remote machines.
- Python API for sending 3D primitives to the browser.
- Python-configurable inputs: buttons, checkboxes, text inputs, sliders,
  dropdowns, gizmos.
- A [meshcat](https://github.com/rdeits/meshcat) and
  [tf](http://wiki.ros.org/tf2)-inspired coordinate frame tree.

The `viser.infra` backend can also be used to build custom web applications
(example:
[the original Nerfstudio viewer](https://github.com/nerfstudio-project/nerfstudio)).
It supports:

- Websocket / HTTP server management, on a shared port.
- Asynchronous server/client communication infrastructure.
- Client state persistence logic.
- Typed serialization; synchronization between Python dataclass and TypeScript
  interfaces.

## Running examples

```bash
# Clone the repository.
git clone https://github.com/nerfstudio-project/viser.git

# Install the package.
# You can also install via pip: `pip install viser`.
cd ./viser
pip install -e .

# Run an example.
pip install -e .[examples]
python ./examples/4_gui.py
```

After an example script is running, you can connect by navigating to the printed
URL (default: `http://localhost:8080`).

See also: our [development docs](https://nerfstudio-project.github.io/viser/development/).
