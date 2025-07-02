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

Examples Gallery
----------------

Install with: ``pip install viser[examples]``

.. include:: examples/_example_gallery.rst


.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 1
   :titlesonly:

   examples_getting_started/index
   examples_scene/index
   examples_gui/index
   examples_interaction/index
   examples_demos/index

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 1
   :titlesonly:

   api_core/index
   api_handles/index
   api_advanced/index
   api_auxiliary/index

.. toctree::
   :caption: Notes
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./conventions.rst
   ./development.rst
   ./embedded_visualizations.rst

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
