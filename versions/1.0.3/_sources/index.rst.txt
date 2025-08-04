Viser
=====

|pyright| |nbsp| |typescript| |nbsp| |versions|

Viser is a 3D visualization library for computer vision and robotics in Python.

Features include:

- API for visualizing 3D primitives
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos)
- Programmatic camera control and rendering
- An entirely web-based client, for easy use over SSH!

The goal is to provide primitives that are (1) easy for simple visualization tasks, but (2) can be composed into more elaborate interfaces. For more about design goals, see the `technical report <https://arxiv.org/abs/2507.22885>`_.

Demo reel:

.. raw:: html

   <video src="https://brentyi.github.io/viser-example-assets/viser_demos.mp4" controls playsinline width="100%"></video><section id="examples">



Examples
--------

Install with: ``pip install viser[examples]``


.. include:: examples/_example_gallery.rst


.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 1
   :titlesonly:

   examples/getting_started/index
   examples/scene/index
   examples/gui/index
   examples/interaction/index
   examples/demos/index

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 1
   :titlesonly:

   api/core/index
   api/handles/index
   api/advanced/index
   api/auxiliary/index

.. toctree::
   :caption: Notes
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./conventions.rst
   ./development.rst
   ./embedded_visualizations.rst
   ./citation.rst

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
