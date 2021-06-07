import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt

from hpss import hpss

def next_pow_2(x):
    return int(2**(np.ceil(np.log2(x))))

def phase_propagation(S, Fs, Ha, Hs, N):
    def p_arg(x):
        return (x + 0.5) % 1.0 - 0.5

    k = np.arange(S.shape[1])
    omega = k * Fs / N

    delta_T = Ha / Fs
    phi = np.angle(S) / (2 * np.pi) + 0.5

    phi_mod = np.copy(phi)
    for m in range(S.shape[0] - 1):
        F_if = omega + p_arg(phi[m+1] - (phi[m] + omega * delta_T)) / delta_T
        phi_mod[m+1] = p_arg(phi_mod[m] + F_if * Hs / Fs) + 0.5

    return phi_mod

def spectrogram(x, fft_size, hop_size, zero_pad=1):
    S = None
    win = np.sqrt(windows.hann(fft_size))
    for i in range(0, len(x), hop_size):
        x_win = np.copy(x[i : i + fft_size])
        x_win *= win[:len(x_win)]
        x_pad = np.zeros(fft_size * zero_pad)
        x_pad[:len(x_win)] = x_win

        if S is None:
            S = np.array([np.fft.fft(x_pad)])
        else:
            S = np.append(S, np.array([np.fft.fft(x_pad)]), axis=0)

    return S

def reconstruct(S, hop_size, window_size, L):
    y = np.zeros(L)
    win = np.sqrt(windows.hann(window_size))
    for i in range(S.shape[0]):
        start_idx = int(i * hop_size)
        n_samples = min(window_size, L - start_idx)
        y[start_idx : start_idx + window_size] += win[:n_samples] * np.real(np.fft.ifft(S[i,:])[:n_samples])
    return y

def stretch(x, fs, stretch_factor):
    window_size_sec = 0.1 # 100 milliseconds
    window_size = next_pow_2(window_size_sec * fs)
    short_window_size = next_pow_2(0.005 * fs)
    
    Hs = window_size // 2 # synthesis hop size
    Ha = int(float(Hs) / stretch_factor)

    Hs_short = short_window_size // 2 # synthesis hop size
    Ha_short = int(float(Hs_short) / stretch_factor)

    print('Computing mono reference phases...')
    x_sum = np.sum(x, axis=0) / 2.0
    X_sum = spectrogram(x_sum, window_size, Ha, zero_pad=1)
    phase_mods = phase_propagation(X_sum, fs, Ha, Hs, window_size)

    stretch_len = int(len(x[0]) * stretch_factor) + 5000
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

        print('\tApplying reference phases...')
        H_full = np.multiply(np.abs(H_full), np.exp(1j * phase_mods))
        P_full = np.multiply(np.abs(P_full), np.exp(1j * phase_mods))

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

        print('\tNormalizing separated signal...')
        h_mag = np.max(np.abs(h_signal))
        p_mag = np.max(np.abs(p_signal))
        h_y *= (h_mag / np.max(np.abs(h_y)))
        p_y *= (p_mag / np.max(np.abs(p_y)))

        y[ch] = h_y + p_y

    # normalize if needed...
    # mag = np.max(np.abs(y))
    # print(f'Original Magnitude {np.max(np.abs(x))}')
    # print(f'Stretched Magnitude {mag}')
    # if mag > 1.0:
    #     y /= mag

    return y
