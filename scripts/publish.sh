#!/bin/bash
set -e

echo "Cleaning dist directory..."
rm -rf dist/*

echo "Building wheels and source distribution..."
uv run maturin build --release --out dist

echo "Uploading to PyPI with twine..."
uv run twine upload dist/*

echo "Publishing complete!"
