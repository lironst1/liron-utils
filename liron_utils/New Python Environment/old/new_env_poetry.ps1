# Exit if any command fails
$ErrorActionPreference = "Stop"

# Install poetry extenstions
poetry self add poetry-plugin-export
poetry self add poetry-plugin-shell

# Create new poetry environment with the current python version (make sure to cd to this script's directory first)
poetry env use python
poetry update
