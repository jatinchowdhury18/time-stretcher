import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from stretch import stretch

START_SECONDS = 125
NUM_SECONDS = 1
TEST_FILE = '/Users/jachowdhury/Downloads/Tennyson - Old Singles/Tennyson - Old Singles - 01 All Yours.wav'
STRETCH_AMOUNT = 1.5

fs, x = wavfile.read(TEST_FILE)
x = (np.transpose(x) / np.max(np.abs(x))).astype(np.float32)

start_idx = int(START_SECONDS * fs)
n_samples = int(NUM_SECONDS * fs)
ref_signal = x[:,start_idx : start_idx + n_samples]
wavfile.write('ref.wav', fs, np.transpose(ref_signal))

stretch_signal = stretch(ref_signal, fs, STRETCH_AMOUNT)

wavfile.write('stretch.wav', fs, np.transpose(stretch_signal))

# plt.plot(ref_signal[0])
# plt.plot(stretch_signal[0])
# plt.show()
