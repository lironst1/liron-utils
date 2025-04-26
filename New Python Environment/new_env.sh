#!/bin/bash

# Exit if any command fails
set -e

# Set environment name
ENV_NAME="myenv"

# Create ~/.condarc file
conda config --set pip_interop_enabled True
conda config --set add_pip_as_python_dependency True

# Don't enable more than one channel as this can cause dependency issues
#conda config --add channels anaconda
#conda config --add channels conda-forge

conda create -y -n "$ENV_NAME" \
  Cython \
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

#conda info
conda activate "$ENV_NAME"

# Upgrade pip
#pip install --upgrade pip

# Install pip-only packages
pip install audioread latex pdflatex librosa manim pipdeptree python-docx pytube pystoi qiskit qutip resampy SoundFile tmm torch torchmetrics uncertainties
#qsharp qsharp-widgets azure-quantum

echo "✅ Conda environment '$ENV_NAME' is ready!"
