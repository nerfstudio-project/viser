# Generating Example Screenshots

The documentation includes automatic screenshot generation for all examples.

## Setup

1. Install playwright:
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Generate screenshots:
   ```bash
   cd docs
   python capture_example_screenshots.py
   ```

## How it Works

The `capture_example_screenshots.py` script:

1. Finds all Python examples in `examples_dev/`
2. Processes examples in parallel batches (default batch size: 4)
3. Runs each example in a subprocess with a unique port
4. Opens a headless browser to connect to each viser server
5. Captures square screenshots (800x800) after the scene loads
6. Saves screenshots to `docs/source/_static/examples/`

## Features

- **Parallel Processing**: Runs multiple examples simultaneously for faster generation
- **Square Screenshots**: All screenshots are 800x800 pixels for consistent display
- **Port Management**: Uses `_VISER_PORT_OVERRIDE` environment variable to run each example on a different port
- **Error Handling**: Continues processing even if some examples fail

## Environment Variables

The script uses the `_VISER_PORT_OVERRIDE` environment variable to assign unique ports to each example in a batch. This prevents port conflicts when running multiple viser servers simultaneously.

## Adding Screenshots to Documentation

Screenshots are displayed using HTML galleries in the RST files:

```rst
.. raw:: html

   <div class="example-gallery">
       <div class="example-card">
           <img src="../../_static/examples/01_scene_00_coordinate_frames.png" alt="Description">
           <div class="example-card-content">
               <h4>00_coordinate_frames.py</h4>
               <p>Description of the example</p>
           </div>
       </div>
   </div>
```

## Troubleshooting

- If screenshots fail to capture, increase the `wait_time` parameter
- Some examples may require additional setup (downloading assets, etc.)
- Examples that require user interaction may not capture well automatically

## CI Integration

To add screenshot generation to CI:

1. Add playwright installation to CI workflow
2. Run `python docs/capture_example_screenshots.py` before building docs
3. Commit generated screenshots or store as artifacts