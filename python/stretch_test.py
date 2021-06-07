import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from stretch import stretch

START_SECONDS = 125
NUM_SECONDS = 10
TEST_FILE = '/Users/jachowdhury/Downloads/Tennyson - Old Singles/Tennyson - Old Singles - 01 All Yours.wav'
STRETCH_AMOUNT = 0.5

fs, x = wavfile.read(TEST_FILE)
x = (np.transpose(x) / np.max(np.abs(x))).astype(np.float32)
start_idx = int(START_SECONDS * fs)
n_samples = int(NUM_SECONDS * fs)
ref_signal = x[:,start_idx : start_idx + n_samples]

# fs = 44100
# N = int(NUM_SECONDS * fs)
# FREQ = 50
# ref_signal = np.zeros((2, N))
# for ch in range(2):
#     ref_signal[ch] = np.sin(2 * np.pi * np.arange(N) * FREQ / fs)

wavfile.write('ref.wav', fs, np.transpose(ref_signal))

stretch_signal = stretch(ref_signal, fs, STRETCH_AMOUNT)

wavfile.write(f'stretch_{STRETCH_AMOUNT}.wav', fs, np.transpose(stretch_signal))

# plt.plot(ref_signal[0])
# plt.plot(np.sin(2 * np.pi * np.arange(len(stretch_signal[0])) * FREQ / fs))
# plt.plot(stretch_signal[0])
# plt.show()
