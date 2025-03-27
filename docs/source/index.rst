viser
=====

|pyright| |nbsp| |typescript| |nbsp| |versions|

**viser** is a library for interactive 3D visualization in Python.

Features include:

- API for visualizing 3D primitives
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos)
- Programmatic camera control and rendering
- An entirely web-based client, for easy use over SSH!

Installation
-----------

You can install ``viser`` with ``pip``:

.. code-block:: bash

   pip install viser

To include example dependencies:

.. code-block:: bash

   pip install viser[examples]

After an example script is running, you can connect by navigating to the printed
URL (default: ``http://localhost:8080``).

.. toctree::
   :caption: Notes
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./conventions.rst
   ./development.rst
   ./embedded_visualizations.rst

.. toctree::
   :caption: API (Basics)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./server.rst
   ./scene_api.rst
   ./gui_api.rst
   ./state_serializer.rst


.. toctree::
   :caption: API (Advanced)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./client_handles.rst
   ./camera_handles.rst
   ./gui_handles.rst
   ./scene_handles.rst
   ./events.rst
   ./icons.rst


.. toctree::
   :caption: API (Auxiliary)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./transforms.rst
   ./infrastructure.rst
   ./extras.rst

.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 1
   :titlesonly:
   :glob:

   examples/*


.. |pyright| image:: https://github.com/nerfstudio-project/viser/actions/workflows/pyright.yml/badge.svg
   :alt: Pyright status icon
   :target: https://github.com/nerfstudio-project/viser
.. |typescript| image:: https://github.com/nerfstudio-project/viser/actions/workflows/typescript-compile.yml/badge.svg
   :alt: TypeScript status icon
   :target: https://github.com/nerfstudio-project/viser
.. |versions| image:: https://img.shields.io/pypi/pyversions/viser
   :alt: Version icon
   :target: https://pypi.org/project/viser/
.. |nbsp| unicode:: 0xA0
   :trim: