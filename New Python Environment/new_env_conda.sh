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

conda create -y -n "$ENV_NAME" -c conda-forge -c pytorch \
  Cython \
  devbio-napari \
  ffmpeg \
  ipykernel \
  ipympl \
  ipython \
  keras \
  matplotlib \
  napari \
  natsort \
  notebook \
  numpy \
  openpyxl \
  pandas \
  pip \
  plotly \
  prettytable \
  pyautogui \
  pycairo \
  pynput \
  PyPDF2 \
  pyqt \
  pywavelets \
  scikit-learn \
  scikit-image \
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
pip install audioread latex pdflatex librosa manim opencv-python pipdeptree pyclesperanto python-docx pytube pystoi qiskit qutip resampy SoundFile tmm torch torchmetrics uncertainties
#qsharp qsharp-widgets azure-quantum

echo "✅ Conda environment '$ENV_NAME' is ready!"
