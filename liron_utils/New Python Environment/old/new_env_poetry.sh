#!/bin/bash

# Exit if any command fails
set -e

# Install poetry extenstions
poetry self add poetry-plugin-export
poetry self add poetry-plugin-shell

# Create new poetry environment with the current python version (make sure to cd to this script's directory first)
poetry env use python
poetry update

echo "✅ Poetry environment is ready!"
