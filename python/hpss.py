import numpy as np

eps = np.finfo(np.float32).eps

def median_filter(x, kernel_size):
    pad = (kernel_size - 1) // 2
    x_pad = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
    y = np.zeros_like(x)
    for i in range(len(y)):
        median_arg = np.argsort(np.abs(x_pad[i : i + kernel_size]))[pad]
        y[i] = x_pad[i + median_arg]

    return y

def hpss(x, perc_kernel=17, harm_kernel=17, mask_power=2, fft_size=4096, hop_size=1024):
    ''' Simple harmonic/percussive source separation based on median filter method '''

    print('Computing HPSS...')
    print('Computing FFT...')
    S = None
    for i in range(0, len(x), hop_size):
        x_win = x[i : i + hop_size]
        x_pad = np.zeros(fft_size)
        x_pad[:len(x_win)] = x_win

        if S is None:
            S = np.array([np.fft.fft(x_pad)])
        else:
            S = np.append(S, np.array([np.fft.fft(x_pad)]), axis=0)
    
    # percussive signal
    print('Separating percussive signal...')
    P = np.copy(S)
    for i in range(S.shape[0]):
        P[i, :] = median_filter(np.abs(S[i, :]), kernel_size=perc_kernel)

    # harmonic signal
    print('Separating harmonic signal...')
    H = np.copy(S)
    for h in range(S.shape[1]):
        H[:, h] = median_filter(np.abs(S[:, h]), kernel_size=harm_kernel)

    # create filter masks
    print('Creating filter masks...')
    M_H = np.copy(S)
    M_P = np.copy(S)
    for i in range(S.shape[0]):
        for h in range(S.shape[1]):
            H_p = H[i,h]**mask_power
            P_p = P[i,h]**mask_power
            denom = H_p + P_p + eps

            M_H[i, h] = H_p / denom
            M_P[i, h] = P_p / denom

    H_hat = np.multiply(S, M_H)
    P_hat = np.multiply(S, M_P)

    print('Computing time-domain signal...')
    h_sig = np.zeros_like(x)
    p_sig = np.zeros_like(x)
    for i in range(S.shape[0]):
        start_idx = int(i * hop_size)
        n_samples = min(hop_size, len(x) - start_idx)
        h_sig[start_idx : start_idx + hop_size] = np.real(np.fft.ifft(H_hat[i,:])[:n_samples])
        p_sig[start_idx : start_idx + hop_size] = np.real(np.fft.ifft(P_hat[i,:])[:n_samples])

    return h_sig, p_sig
