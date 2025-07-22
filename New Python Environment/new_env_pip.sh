#!/bin/bash

# Exit if any command fails
set -e

# Set environment name and location
ENV_NAME="myenv"

# Create the virtual environment
#python -m venv "$ENV_NAME"

# Activate the environment
source "$ENV_NAME/Scripts/activate"

# Install all packages via pip
python -m pip install -r "pip_requirements.txt" -U --progress-bar on

echo "✅ Virtual environment '$ENV_NAME' is ready!"
