# Viser Examples (Development)

This is the development version of examples for `viser`. This should be compatible with:

- The latest development version of `viser` on GitHub.

But not necessarily:

- The latest release of `viser` on PyPI.

## Quick Start

Start with `00_hello_world.py` - the simplest possible viser program:

```bash
python 00_hello_world.py
```

Then open your browser to `http://localhost:8080` to see your first 3D visualization!

## Example Categories

Our examples are organized by functionality to help you find what you need:

### 🎯 [01_scene/](01_scene/) - 3D Visualization Basics
Learn the fundamentals: coordinate systems, meshes, cameras, and lighting.
Perfect for getting started with 3D graphics concepts.

### 🎛️ [02_gui/](02_gui/) - User Interface
Build interactive control panels with buttons, sliders, and custom layouts.
Essential for creating user-friendly applications.

### 🖱️ [03_interaction/](03_interaction/) - User Input & Events  
Handle mouse clicks, scene picking, and real-time interaction.
Great for building interactive tools and applications.

### 🚀 [04_demos/](04_demos/) - Complete Applications
Real-world examples integrating external tools like RealSense, COLMAP, and URDF.
See viser in action with practical use cases.

## Learning Path

1. **Start here**: `00_hello_world.py` - Your first viser program
2. **Learn basics**: Explore `01_scene/` examples (coordinate frames, meshes, cameras)
3. **Add interaction**: Try `02_gui/` examples (controls, layouts, theming)
4. **Go interactive**: Experiment with `03_interaction/` examples (clicks, events)
5. **See it in action**: Run `04_demos/` for complete applications

## Running Examples

Most examples can be run directly:

```bash
python 01_scene/00_coordinate_frames.py
python 02_gui/00_basic_controls.py
python 03_interaction/00_click_meshes.py
```

Some demos require additional data - check the download scripts in `assets/`:

```bash
./assets/download_dragon_mesh.sh
python 01_scene/02_meshes.py
```

## Getting Help

- Each directory has its own README with detailed explanations
- Example files include comprehensive docstrings
- Check the main [viser documentation](https://viser.studio) for API reference
