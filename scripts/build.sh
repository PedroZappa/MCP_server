#!/usr/bin/env bash

CWD=$(pwd)
echo "Working in directory: $CWD"

# Create and activate virtual environment
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo ".venv created"
fi

source .venv/bin/activate
pip install --upgrade pip

# Priority order: pyproject.toml > requirements.txt > setup.py
if [ -f "pyproject.toml" ]; then
    echo "Installing from pyproject.toml..."
    if grep -q "optional-dependencies" pyproject.toml; then
        pip install -e .[dev]
    else
        pip install -e .
    fi
elif [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
elif [ -f "setup.py" ]; then
    echo "Installing from setup.py..."
    pip install -e .
else
    echo "No dependency configuration found in $CWD"
    exit 1
fi

pip list

