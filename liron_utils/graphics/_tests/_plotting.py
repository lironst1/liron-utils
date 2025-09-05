import numpy as np
from scipy.signal import firwin2, butter
from scipy.signal import windows

from liron_utils import graphics as gr
from liron_utils.signal_processing import analyze_window, analyze_filter

b = firwin2(numtaps=21, freq=[0, 0.1, 0.2, 1], gain=[1, 1, 0, 0])
# b = windows.get_window("hamming", 6)
# b /= b.sum()
a = 1

# b, a = butter(N=8, Wn=[0.3, 0.7], btype="bandpass")

# out = analyze_window(b, fs=None, plot=True)
out = analyze_filter(b, a, plot=True)

# Ax = gr.Axes(shape=(3, 1))
# Ax[0, 0].plot_impulse_response(b, a, n=50)
# Ax[1, 0].plot_frequency_response(b, a, one_sided=False, which="amp")
# Ax[2, 0].plot_frequency_response(b, a, one_sided=False, which="phase")
# Ax.set_props()
