#!/bin/bash

# Exit if any command fails
set -e

# Set environment name and location
ENV_NAME="myenv"

# Create the virtual environment
#python -m venv "$ENV_NAME"

# Activate the environment
source "$ENV_NAME/Scripts/activate"

# Upgrade pip, setuptools, and wheel
C:\\Users\\lironst\\myenv\\Scripts\\python.exe -m pip install --upgrade pip setuptools wheel

# Install all packages via pip
pip install \
  Cython \
  devbio-napari \
  ffmpeg-python \
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
  plotly \
  prettytable \
  pyautogui \
  pycairo \
  pynput \
  PyPDF2 \
  pyqt5 \
  scikit-learn \
  scikit-image \
  scipy \
  selenium \
  tensorboard \
  tensorflow \
  tqdm \
  audioread \
  latex \
  pdflatex \
  librosa \
  manim \
  opencv-python \
  pipdeptree \
  pyclesperanto-prototype \
  python-docx \
  pytube \
  pystoi \
  qiskit \
  qutip \
  resampy \
  SoundFile \
  tmm \
  torch \
  torchmetrics \
  uncertainties

echo "✅ Virtual environment '$ENV_NAME' is ready!"
