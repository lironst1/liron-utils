#!/bin/bash

# Exit if any command fails
set -e

# Set environment name
PYTHON_VERSION="3.11.14"
ENV_NAME="myvenv"

pyenv virtualenv "$PYTHON_VERSION" "$ENV_NAME"  # Change to desired version
pyenv global "$PYTHON_VERSION/envs/myvenv"

poetry update

poetry export -f requirements.txt --output requirements.txt --without-hashes