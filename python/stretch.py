import numpy as np
from scipy.signal import windows

from hpss import hpss, spectrogram

def next_pow_2(x):
    return int(2**(np.ceil(np.log2(x))))

def reconstruct(S, hop_size, window_size, L):
    y = np.zeros(L)
    for i in range(S.shape[0]):
        start_idx = int(i * hop_size)
        n_samples = min(window_size, L - start_idx)
        win = windows.hann(window_size)[:n_samples]
        y[start_idx : start_idx + window_size] += win * np.real(np.fft.ifft(S[i,:])[:n_samples])
    return y

def stretch(x, fs, stretch_factor):
    window_size_sec = 0.1 # 100 milliseconds
    window_size = next_pow_2(window_size_sec * fs)
    short_window_size = next_pow_2(0.01 * fs)
    
    Hs = window_size // 2 # synthesis hop size
    Ha = int(float(Hs) / stretch_factor)

    Hs_short = short_window_size // 2 # synthesis hop size
    Ha_short = int(float(Hs_short) / stretch_factor)

    print('Computing mono reference phases...')
    x_sum = np.sum(x, axis=0)
    X_sum = spectrogram(x_sum, window_size, Ha, zero_pad=1)
    # @TODO: compute phase vocoder phases

    stretch_len = int(len(x[0]) * stretch_factor) + 1000
    y = np.zeros((2, stretch_len))
    for ch in range(x.shape[0]):
        print(f'Processing channel {ch}...')
        h_signal, p_signal = hpss(x[ch])

        print('Performing time-stretching...')
        H_full = spectrogram(h_signal, window_size, Ha)
        P_full = spectrogram(p_signal, window_size, Ha)

        print('\tSeparated magnitude-only PV...')
        h_x_long = reconstruct(H_full, Hs, window_size, stretch_len)
        P_short = spectrogram(p_signal, short_window_size, Ha_short)
        p_x_short = reconstruct(P_short, Hs_short, short_window_size, stretch_len)

        print('\tComputing reference phases...')
        # @TODO: correct full spectrograms with reference phases

        print('\tReconstructing references...')
        h_v = reconstruct(H_full, Hs, window_size, stretch_len)
        p_v = reconstruct(P_full, Hs, window_size, stretch_len)

        print('\tPerforming magnitude correction...')
        H_v_long = spectrogram(h_v, window_size, Ha)
        P_v_short = spectrogram(p_v, short_window_size, Ha_short)
        
        H_w_long = spectrogram(h_x_long, window_size, Ha)
        P_w_short = spectrogram(p_x_short, short_window_size, Ha_short)

        H_y = np.multiply(np.abs(H_w_long), np.exp(1j * np.angle(H_v_long)))
        P_y = np.multiply(np.abs(P_w_short), np.exp(1j * np.angle(P_v_short)))

        print('\tReconstructing final signal...')
        h_y = reconstruct(H_y, Ha, window_size, stretch_len)
        p_y = reconstruct(P_y, Ha_short, short_window_size, stretch_len)

        y[ch] = h_y + p_y

    return y
