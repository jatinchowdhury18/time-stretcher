#include "hpss.h"
#include "fft_utils.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace HPSS
{

using namespace fft_utils;
using Vec2D = std::vector<std::vector<float>>;

/** Performs "horizontal" median filtering to obtain harmonic signal */
Vec2D median_filter_harm(const std::vector<fftw_complex_vec>& S, int kernel_size)
{
    Vec2D H (S.size(), std::vector<float> (S[0].size(), 0.0f));

    const int pad = (kernel_size - 1) / 2;
    std::vector<float> med_vec (S.size() + 2 * pad, 0.0f);
    std::vector<float> kernel_vec (kernel_size, 0.0f);

    for(int h = 0; h < (int) S[0].size(); ++h)
    {
        for(int i = 0; i < (int) S.size(); ++i)
            med_vec[i + pad] = std::abs(S[i][h]);

        for(int i = 0; i < (int) S.size(); ++i)
        {
            std::copy(med_vec.begin() + i, med_vec.begin() + i + kernel_size, kernel_vec.begin());
            std::nth_element(kernel_vec.begin(), kernel_vec.begin() + pad, kernel_vec.end());
            H[i][h] = kernel_vec[pad];
        }
    }

    return H;
}

/** Performs "vertical" median filtering to obtain percussive signal */
Vec2D median_filter_perc(const std::vector<fftw_complex_vec>& S, int kernel_size)
{
    Vec2D P (S.size(), std::vector<float> (S[0].size(), 0.0f));

    const int pad = (kernel_size - 1) / 2;
    std::vector<float> med_vec (S[0].size() + 2 * pad, 0.0f);
    std::vector<float> kernel_vec (kernel_size, 0.0f);

    for(int i = 0; i < (int) S.size(); ++i)
    {
        for(int h = 0; h < (int) S[i].size(); ++h)
            med_vec[h + pad] = std::abs(S[i][h]);

        for(int h = 0; h < (int) S[i].size(); ++h)
        {
            std::copy(med_vec.begin() + i, med_vec.begin() + i + kernel_size, kernel_vec.begin());
            std::nth_element(kernel_vec.begin(), kernel_vec.begin() + pad, kernel_vec.end());
            P[i][h] = kernel_vec[pad];
        }
    }

    return P;
}

/** Computes a time-frequency spectrogram with no window (for now) */
inline std::vector<fftw_complex_vec> spectrogram(const std::vector<float>& x, int fft_size, int hop_size, int zero_pad = 1)
{
    const auto n_fft = fft_size * zero_pad;

    std::vector<fftw_complex_vec> S;
    fft_utils::ForwardFFT fft { n_fft };
    for(int i = 0; i + fft_size < (int) x.size(); i += hop_size)
    {
        std::copy(x.begin() + i, x.begin() + i + fft_size, fft.x_in.data());
        fft.perform();
        S.push_back(fft.Y_out);
    }

    return S;
}

/** Reconstructs the harmonic and percussive signals from their spectrograms with a Hann window */
std::pair<std::vector<float>, std::vector<float>> spec_reconstruct(std::vector<fftw_complex_vec>& H_hat,
                                                                   std::vector<fftw_complex_vec>& P_hat,
                                                                   int n_samples,
                                                                   const int fft_size,
                                                                   const int hop_size)
{
    std::vector<float> h_sig (n_samples, 0.0f);
    std::vector<float> p_sig (n_samples, 0.0f);
    const auto win = hann(fft_size, float(fft_size / hop_size) / 2.0f);

    const int n_fft = (int) H_hat[0].size();
    fft_utils::InverseFFT ifft { n_fft };
    for(int i = 0; i < H_hat.size(); ++i)
    {
        int start_idx = i * hop_size;
        int samples = std::min(fft_size, n_samples - start_idx);

        { // do H
            std::copy(H_hat[i].begin(), H_hat[i].end(), ifft.X_in.data());
            ifft.perform();
            fft_utils::applyWindow(ifft.y_out, win, ifft.y_out);

            for(int n = 0; n < samples; ++n)
                h_sig[n + start_idx] += ifft.y_out[n];
        }

        { // do P
            std::copy(P_hat[i].begin(), P_hat[i].end(), ifft.X_in.data());
            ifft.perform();
            fft_utils::applyWindow(ifft.y_out, win, ifft.y_out);

            for(int n = 0; n < samples; ++n)
                p_sig[n + start_idx] += ifft.y_out[n];
        }
    }

    return std::make_pair(h_sig, p_sig);
}

static void debug_print(const std::string& str, bool debug)
{
    if(debug)
        std::cout << str << std::endl;
}

std::pair<std::vector<float>, std::vector<float>> hpss(std::vector<float> x, const HPSS_PARAMS& params)
{
    const auto fft_size = next_pow2(int(params.window_length_ms * 0.001 * params.sample_rate));
    const auto hop_size = next_pow2(int(params.hop_length_ms * 0.001 * params.sample_rate));

    debug_print("Computing HPSS...", params.debug);
    debug_print("\tComputing STFTs...", params.debug);
    auto S = spectrogram(x, fft_size, hop_size, params.zero_pad);

    debug_print("\tSeparating percussive signal...", params.debug);
    auto P = median_filter_perc(S, params.perc_kernel);

    debug_print("\tSeparating harmonic signal...", params.debug);
    auto H = median_filter_harm(S, params.harm_kernel);

    debug_print("\tApplying filter masks...", params.debug);
    std::vector<fftw_complex_vec> H_hat (S.size(), fftw_complex_vec(S[0].size()));
    std::vector<fftw_complex_vec> P_hat (S.size(), fftw_complex_vec(S[0].size()));
    for(int i = 0; i < (int) S.size(); ++i)
    {
        for(int h = 0; h < (int) S[0].size(); ++h)
        {
            float H_p = std::pow(H[i][h], params.mask_exp);
            float P_p = std::pow(P[i][h], params.mask_exp);
            float denom = H_p + P_p + std::numeric_limits<float>::epsilon();

            H_hat[i][h] = S[i][h] * (H_p / denom);
            P_hat[i][h] = S[i][h] * (P_p / denom);
        }
    }

    debug_print("\tComputing time-domain signal...", params.debug);
    return spec_reconstruct(H_hat, P_hat, (int) x.size(), fft_size, hop_size);
}

} // namespace HPSS

