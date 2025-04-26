#!/bin/bash

# Exit if any command fails
set -e

# Set environment name
ENV_NAME="myenv"

#export MAMBA_EXE="$HOME/.local/bin/micromamba"
# export MAMBA_ROOT_PREFIX="C:/ProgramData/miniconda3/envs"
#eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"

# Create and activate environment
micromamba create -y -v -n "$ENV_NAME" -c defaults -c conda-forge \
  cython \
  ffmpeg \
  ipykernel \
  ipympl \
  ipython \
  keras \
  matplotlib \
  notebook \
  numpy \
  opencv \
  pandas \
  pip \
  plotly \
  pycairo \
  PyPDF2 \
  scikit-learn \
  scipy \
  selenium \
  tensorboard \
  tensorflow \
  tqdm

micromamba activate "$ENV_NAME"

# Upgrade pip
pip install --upgrade pip

# Install pip-only packages
pip install \
audioread \
latex pdflatex \
librosa \
manim \
pipdeptree \
python-docx \
pytube \
pystoi \
qiskit \
qutip \
qsharp qsharp-widgets azure-quantum \
resampy \
SoundFile \
tmm \
torch torchmetrics \
uncertainties

echo "✅ Micromamba environment '$ENV_NAME' is ready!"
