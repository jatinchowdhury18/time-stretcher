import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from hpss import hpss

START_SECONDS = 125
NUM_SECONDS = 5
TEST_FILE = '/Users/jachowdhury/Downloads/Tennyson - Old Singles/Tennyson - Old Singles - 01 All Yours.wav'

fs, x = wavfile.read(TEST_FILE)
x = (np.transpose(x) / np.max(np.abs(x))).astype(np.float32)

start_idx = int(START_SECONDS * fs)
n_samples = int(NUM_SECONDS * fs)
ref_signal = x[:,start_idx : start_idx + n_samples]
wavfile.write('ref.wav', fs, np.transpose(ref_signal))

h_signal = np.zeros_like(ref_signal)
p_signal = np.zeros_like(ref_signal)
h_signal[0], p_signal[0] = hpss(ref_signal[0])
h_signal[1], p_signal[1] = hpss(ref_signal[1])

wavfile.write('harmonic.wav', fs, np.transpose(h_signal))
wavfile.write('percussive.wav', fs, np.transpose(p_signal))
wavfile.write('sum.wav', fs, np.transpose(h_signal + p_signal))
