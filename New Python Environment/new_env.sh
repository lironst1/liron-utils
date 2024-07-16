##create ~/.condarc file
conda config

conda config --set pip_interop_enabled True
conda config --set add_pip_as_python_dependency True

##don't enable more than one channel as this can cause dependency issues
#conda config --add channels anaconda
#conda config --add channels conda-forge

conda config --add create_default_packages pip
conda config --add create_default_packages numpy
conda config --add create_default_packages scipy
conda config --add create_default_packages scikit-learn
conda config --add create_default_packages Cython
conda config --add create_default_packages ffmpeg
conda config --add create_default_packages keras
conda config --add create_default_packages matplotlib
conda config --add create_default_packages notebook
conda config --add create_default_packages opencv
conda config --add create_default_packages ipython
conda config --add create_default_packages pandas
conda config --add create_default_packages plotly
conda config --add create_default_packages pycairo
conda config --add create_default_packages PyPDF2
conda config --add create_default_packages selenium
conda config --add create_default_packages tensorboard
conda config --add create_default_packages tensorflow
conda config --add create_default_packages tqdm

#conda info

conda create --name myenv

pip install --upgrade pip
pip install \
audioread \
ipympl \
ipykernel \
latex \
librosa \
manim \
pdflatex \
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
torch \
torchmetrics \
uncertainties \
