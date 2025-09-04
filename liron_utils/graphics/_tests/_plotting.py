import numpy as np
from liron_utils import graphics as gr
from scipy.signal import firwin2, butter, filtfilt

gr.update_rcParams("liron-utils-text-color", "white")

b = firwin2(numtaps=50, freq=[0, 0.19, 0.21, 1], gain=[1, 1, 0, 0])
a = 1

# b, a = butter(N=4, Wn=0.8)

Ax = gr.Axes(shape=(3, 1))
Ax[0, 0].plot_impulse_response(b, a, dt=1, n=500)
Ax[1, 0].plot_frequency_response(b, a, one_sided=False, which="amp")
Ax[2, 0].plot_frequency_response(b, a, one_sided=False, which="phase")
Ax.set_props()
