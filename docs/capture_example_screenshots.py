#!/usr/bin/env python3
"""
Automatically run viser examples and capture screenshots for documentation.

This script:
1. Runs each example in a subprocess
2. Opens a browser to connect to the viser server
3. Captures a screenshot
4. Saves it to docs/source/_static/examples/
"""

import asyncio
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from playwright.async_api import async_playwright

except ImportError:
    raise SystemExit(
        "Warning: playwright not installed. Install with: pip install playwright && playwright install chromium"
    )


async def capture_screenshot_playwright(
    url: str, output_path: Path, wait_time: float = 3.0
) -> bool:
    """Capture screenshot using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # Set viewport to 16:9 aspect ratio with higher resolution for better quality
        # Using 2x resolution for retina-quality screenshots
        page = await browser.new_page(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=2  # This gives us 2x DPI
        )

        try:
            # Navigate to the page
            await page.goto(url)

            # Wait for the scene to load
            await asyncio.sleep(wait_time)

            # Take screenshot - it will be 3840x2160 due to device_scale_factor
            await page.screenshot(
                path=str(output_path),
                type="png"
            )

            print(f"  ✓ Screenshot saved to {output_path}")
            return True

        except Exception as e:
            print(f"  ✗ Failed to capture screenshot: {e}")
            return False
        finally:
            await browser.close()


def extract_title_and_description(file_path: Path) -> Tuple[str, str]:
    """Extract title (first line) and description from module docstring."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the Python file to extract docstring
        import ast
        tree = ast.parse(content)
        
        # Get module docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            lines = docstring.strip().split('\n')
            if lines:
                # First line is the title
                title = lines[0].strip()
                
                # Rest is description - find first non-empty line after title
                description = ""
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        # Get first sentence of description
                        sentences = line.split('.')
                        if sentences and sentences[0]:
                            description = sentences[0].strip() + "."
                        else:
                            description = line
                        break
                
                return title, description
        
        # Fallback to filename-based
        title = file_path.stem.replace("_", " ").title()
        return title, ""
    except Exception:
        title = file_path.stem.replace("_", " ").title()
        return title, ""


def find_examples() -> List[Tuple[str, Path, str, str]]:
    """Find all Python examples in the examples_dev directory with titles and descriptions."""
    examples = []
    examples_dir = Path(__file__).parent.parent / "examples_dev"

    # Add hello world example
    hello_world = examples_dir / "00_hello_world.py"
    if hello_world.exists():
        title, desc = extract_title_and_description(hello_world)
        examples.append(("00_hello_world", hello_world, title, desc))

    # Add examples from each category
    categories = ["01_scene", "02_gui", "03_interaction", "04_demos"]
    for category in categories:
        category_dir = examples_dir / category
        if category_dir.exists():
            for example_file in sorted(category_dir.glob("*.py")):
                if not example_file.name.startswith("_"):  # Skip private files
                    example_name = f"{category}/{example_file.stem}"
                    title, desc = extract_title_and_description(example_file)
                    examples.append((example_name, example_file, title, desc))

    return examples


async def run_example_and_capture(
    example_name: str, example_path: Path, output_dir: Path, port: int
) -> Tuple[str, bool]:
    """Run a single example and capture its screenshot."""
    print(f"Processing {example_name} on port {port}...")

    # Create output filename
    safe_name = example_name.replace("/", "_")
    output_path = output_dir / f"{safe_name}.png"

    # Start the example process with specific port using environment variable
    env = os.environ.copy()
    env["_VISER_PORT_OVERRIDE"] = str(port)

    process = subprocess.Popen(
        [sys.executable, str(example_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        # Wait a bit for the server to start
        await asyncio.sleep(2.0)

        # Check if process is still running
        if process.poll() is not None:
            print(f"  ✗ {example_name} exited early with code {process.returncode}")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"  Error: {stderr}")
            return example_name, False

        # Capture screenshot
        success = await capture_screenshot_playwright(
            f"http://localhost:{port}", output_path
        )

        if success:
            print(f"  ✓ {example_name} completed")
        return example_name, success

    finally:
        # Clean up the process
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def process_batch(
    batch: List[Tuple[str, Path, str, str]], output_dir: Path
) -> List[Tuple[str, bool]]:
    """Process a batch of examples in parallel with random ports."""
    tasks = []
    used_ports = set()
    
    for example_name, example_path, _title, _desc in batch:
        # Generate a random port between 8080 and 9999
        while True:
            port = random.randint(8080, 9999)
            if port not in used_ports:
                used_ports.add(port)
                break
        
        task = run_example_and_capture(example_name, example_path, output_dir, port)
        tasks.append(task)

    return await asyncio.gather(*tasks)


async def capture_all_screenshots(batch_size: int = 8):
    """Main function to capture screenshots for all examples."""
    # Create output directory
    output_dir = Path(__file__).parent / "source" / "_static" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all examples
    examples = find_examples()
    print(f"Found {len(examples)} examples to process")
    print(f"Processing in batches of {batch_size}")

    # Process examples in batches
    successful = 0
    failed = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        print(
            f"\nBatch {i // batch_size + 1}/{(len(examples) + batch_size - 1) // batch_size}"
        )

        # Process batch in parallel
        results = await process_batch(batch, output_dir)

        # Count successes and failures
        for example_name, success in results:
            if success:
                successful += 1
            else:
                failed.append(example_name)

        # Small delay between batches
        if i + batch_size < len(examples):
            await asyncio.sleep(1.0)

    # Summary
    print("\n" + "=" * 50)
    print("Screenshot capture complete!")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {len(failed)}")
    if failed:
        print(f"  Failed examples: {', '.join(failed)}")

    # Generate RST include file and code pages
    generate_screenshot_includes(examples, output_dir)
    generate_code_pages(examples, output_dir)


def generate_code_pages(examples: List[Tuple[str, Path, str, str]], output_dir: Path):
    """Generate RST pages for each example with syntax-highlighted code."""
    examples_rst_dir = output_dir.parent.parent / "examples" / "generated"
    examples_rst_dir.mkdir(exist_ok=True)
    
    for example_name, example_path, title, desc in examples:
        # Create RST filename
        safe_name = example_name.replace("/", "_")
        rst_path = examples_rst_dir / f"{safe_name}.rst"
        
        # Read the source code
        with open(example_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Write RST file with code
        with open(rst_path, "w") as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            
            if desc:
                f.write(f"{desc}\n\n")
            
            f.write(f"**Source:** ``examples_dev/{example_name}.py``\n\n")
            
            # Add screenshot if it exists
            screenshot_path = f"../_static/examples/{safe_name}.png"
            f.write(".. figure:: " + screenshot_path + "\n")
            f.write("   :width: 100%\n")
            f.write("   :alt: " + title + "\n\n")
            
            f.write("Code\n")
            f.write("----\n\n")
            f.write(".. code-block:: python\n")
            f.write("   :linenos:\n\n")
            
            # Indent the code
            for line in source_code.split('\n'):
                f.write(f"   {line}\n")
    
    print(f"Generated {len(examples)} code pages in {examples_rst_dir}")
    
    # Generate index file for the generated examples
    index_path = examples_rst_dir / "index.rst"
    with open(index_path, "w") as f:
        f.write("Example Code\n")
        f.write("============\n\n")
        f.write(".. This file is auto-generated by capture_example_screenshots.py\n\n")
        f.write(".. toctree::\n")
        f.write("   :hidden:\n")
        f.write("   :maxdepth: 1\n\n")
        
        for example_name, _, _, _ in sorted(examples):
            safe_name = example_name.replace("/", "_")
            f.write(f"   {safe_name}\n")


def generate_screenshot_includes(examples: List[Tuple[str, Path, str, str]], output_dir: Path):
    """Generate RST include file for screenshots with a nice gallery."""
    include_file = output_dir.parent.parent / "examples" / "_example_gallery.rst"

    # Category metadata
    categories = {
        "00_hello_world": ("👋 Getting Started", 0),
        "01_scene": ("🎯 Scene Fundamentals", 1),
        "02_gui": ("🎛️ GUI Controls", 2),
        "03_interaction": ("🖱️ User Interaction", 3),
        "04_demos": ("🚀 Complete Applications", 4),
    }

    # Group examples by category
    grouped = {}
    for example_name, example_path, title, desc in examples:
        if "/" in example_name:
            category = example_name.split("/")[0]
        else:
            category = example_name
        
        if category not in grouped:
            grouped[category] = []
        grouped[category].append((example_name, example_path, title, desc))

    with open(include_file, "w") as f:
        f.write(".. This file is auto-generated by capture_example_screenshots.py\n")
        f.write(".. Include this in your documentation with: .. include:: _example_gallery.rst\n\n")
        
        # Sort categories by their order
        sorted_categories = sorted(grouped.keys(), key=lambda x: categories.get(x, (x, 999))[1])

        # Write gallery content
        for category in sorted_categories:
            category_name, _ = categories.get(category, (category.replace("_", " ").title(), 999))
            
            f.write(f"\n{category_name}\n")
            f.write("~" * len(category_name) + "\n\n")
            
            # Create a grid table for examples
            f.write(".. raw:: html\n\n")
            f.write('   <style>\n')
            f.write('   .example-card:hover { transform: translateY(-4px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }\n')
            f.write('   </style>\n')
            f.write('   <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; margin: 20px 0;">\n')
            
            for example_name, example_path, title, desc in grouped[category]:
                safe_name = example_name.replace("/", "_")
                image_path = f"_static/examples/{safe_name}.png"
                code_page_url = f"generated/{safe_name}.html"
                
                # Clean text for HTML: remove line breaks and escape quotes
                title_clean = title.replace('\n', ' ').replace('\r', ' ').replace('"', '&quot;').strip()
                desc_clean = desc.replace('\n', ' ').replace('\r', ' ').replace('"', '&quot;').strip()
                
                f.write('       <div class="example-card" style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white; transition: transform 0.2s;">\n')
                f.write(f'           <a href="{code_page_url}" style="text-decoration: none; color: inherit; display: block;">\n')
                f.write(f'               <img src="{image_path}" alt="{title_clean}" style="width: 100%; height: auto; aspect-ratio: 16/9; object-fit: cover; display: block;">\n')
                f.write('               <div style="padding: 15px;">\n')
                f.write(f'                   <h4 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600; color: #333;">{title_clean}</h4>\n')
                if desc_clean:
                    f.write(f'                   <p style="margin: 0; color: #666; font-size: 13px; line-height: 1.4;">{desc_clean}</p>\n')
                f.write('               </div>\n')
                f.write('           </a>\n')
                f.write('       </div>\n')
            
            f.write('   </div>\n')

    print(f"\nGenerated example gallery at {include_file}")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: This script requires Python 3.7 or later")
        sys.exit(1)

    # Run the main function
    asyncio.run(capture_all_screenshots())
