# viser

|mypy| |nbsp| |pyright| |nbsp| |typescript| |nbsp| |versions|

**viser** is a library for interactive 3D visualization in Python.

Features include:

- API for visualizing 3D primitives
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos)
- Programmatic camera control and rendering
- An entirely web-based client, for easy use over SSH!

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

<!-- prettier-ignore-start -->

.. toctree::
   :caption: Notes
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./conventions.md
   ./development.md
   ./embedded_visualizations.rst

.. toctree::
   :caption: API (Basics)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./server.md
   ./scene_api.md
   ./gui_api.md


.. toctree::
   :caption: API (Advanced)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./client_handles.md
   ./camera_handles.md
   ./gui_handles.md
   ./scene_handles.md
   ./events.md
   ./icons.md


.. toctree::
   :caption: API (Auxiliary)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./transforms.md
   ./infrastructure.md
   ./extras.md

.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 1
   :titlesonly:
   :glob:

   examples/*


.. |build| image:: https://github.com/nerfstudio-project/viser/workflows/build/badge.svg
   :alt: Build status icon
   :target: https://github.com/nerfstudio-project/viser
.. |mypy| image:: https://github.com/nerfstudio-project/viser/workflows/mypy/badge.svg?branch=main
   :alt: Mypy status icon
   :target: https://github.com/nerfstudio-project/viser
.. |pyright| image:: https://github.com/nerfstudio-project/viser/workflows/pyright/badge.svg?branch=main
   :alt: Mypy status icon
   :target: https://github.com/nerfstudio-project/viser
.. |typescript| image:: https://github.com/nerfstudio-project/viser/workflows/typescript-compile/badge.svg
   :alt: TypeScript status icon
   :target: https://github.com/nerfstudio-project/viser
.. |versions| image:: https://img.shields.io/pypi/pyversions/viser
   :alt: Version icon
   :target: https://pypi.org/project/viser/
.. |nbsp| unicode:: 0xA0
   :trim:

<!-- prettier-ignore-end -->
