#include "stretch.h"
#include "fft_utils.h"
#include <cmath>
#include <iostream>

namespace time_stretch
{

using namespace fft_utils;

/** Finds the next power of two larger than the given number */
int next_pow2(int x)
{
    return (int) std::pow(2.0f, std::ceil(std::log2((double) x)));
}

/** Computes the correct phase propagation for a phase vocoder */
std::vector<std::vector<float>> phase_propagation(const std::vector<fftw_complex_vec>& S, float fs, int Ha, int Hs, int N)
{
    auto p_arg = [] (float x) { return std::fmod(x + 0.5f, 1.0f) - 0.5f; };

    const int M = S.size();
    const int K = S[0].size();
    std::vector<float> omega (K, 0.0f);
    for(int k = 0; k < (int) omega.size(); ++k)
        omega[k] = (float) k * fs / (float) N;

    std::vector<std::vector<float>> phi (M, std::vector<float> (K, 0.0f));
    for(int m = 0; m < M; ++m)
        for(int k = 0; k < K; ++k)
            phi[m][k] = std::arg(S[m][k]) / (2.0f * M_PI) + 0.5f;

    const float delta_T = (float) Ha / fs;
    std::vector<std::vector<float>> phi_mod (M, std::vector<float> (K, 0.0f));
    std::copy(phi[0].begin(), phi[0].end(), phi_mod[0].begin());
    for(int m = 0; m < M - 1; ++m)
    {
        for(int k = 0; k < K; ++k)
        {
            auto F_if = omega[k] + p_arg(phi[m+1][k] - (phi[m][k] + omega[k] * delta_T)) / delta_T;
            phi_mod[m+1][k] = p_arg(phi_mod[m][k] + F_if * (float) Hs / fs) + 0.5f;
        }
    }

    return phi_mod;
}

/** Generates a square-root Hann window */
std::vector<float> sqrt_hann(int N)
{
    auto win = fft_utils::hann(N);
    for(int n = 0; n < N; ++n)
        win[n] = std::sqrt(win[n]);

    return win;
}

/** Computes a time-frequency spectrogram with a sqrt Hann window */
std::vector<fftw_complex_vec> spectrogram(const std::vector<float>& x, int fft_size, int hop_size, int zero_pad = 1)
{
    const auto n_fft = fft_size * zero_pad;
    auto win = sqrt_hann(fft_size);

    std::vector<fftw_complex_vec> S;
    fft_utils::ForwardFFT fft { n_fft };
    for(int i = 0; i + fft_size < (int) x.size(); i += hop_size)
    {
        std::copy(&x[i], &x[i + fft_size], fft.x_in.data());
        fft_utils::applyWindow(fft.x_in, win, fft.x_in);
        fft.perform();
        S.push_back(fft.Y_out);
    }

    return S;
}

/** Reconstructs the signal from its spectrograms with a sqrt Hann window */
std::vector<float> reconstruct(std::vector<fftw_complex_vec>& S, int hop_size, int window_size, int L)
{
    std::vector<float> y (L, 0.0f);
    int final_idx = 0;
    auto win = sqrt_hann(window_size);

    const auto n_fft = (int) S[0].size();
    fft_utils::InverseFFT ifft { n_fft };
    for(int i = 0; i < (int) S.size(); ++i)
    {
        const auto start_idx = int(i * hop_size);
        const auto n_samples = std::min(window_size, L - start_idx);

        std::copy(&S[i][0], &S[i][n_fft], ifft.X_in.data());
        ifft.perform();
        fft_utils::applyWindow(ifft.y_out, win, ifft.y_out);

        for(int n = 0; n < n_samples; ++n)
            y[n + start_idx] += ifft.y_out[n];
        final_idx = start_idx + n_samples;
    }

    return { &y[0], &y[final_idx] };
}

/** Compares greater/less than with absolute value */
template<typename T>
static bool abs_compare(T a, T b)
{
    return (std::abs(a) < std::abs(b));
}

/** Returns the maximum absolute value in a vector */
static float max_abs(const std::vector<float>& vec)
{
    return std::abs(*std::max_element(vec.begin(), vec.end(), abs_compare<float>));
}

/** Normalizes the signal in a vector to have the same magnitude as the reference */
static void normalize_vec(const std::vector<float>& vec_ref, std::vector<float>& vec_cur)
{
    auto mag_ref = max_abs(vec_ref);
    auto mag_cur = max_abs(vec_cur);
    for(int i = 0; i < (int) vec_cur.size(); ++i)
        vec_cur[i] *= mag_ref / mag_cur;
}

static void debug_print(const std::string& str, bool debug)
{
    if(debug)
        std::cout << str << std::endl;
}

std::vector<std::vector<float>> time_stretch(const std::vector<std::vector<float>>& x, const STRETCH_PARAMS& params)
{
    const auto long_window_size = next_pow2(int(params.long_window_ms * 0.001 * params.sample_rate));
    const auto short_window_size = next_pow2(int(params.short_window_ms * 0.001 * params.sample_rate));

    const auto Hs_long = long_window_size / 2;
    const auto Ha_long = int((float) Hs_long / params.stretch_factor);

    const auto Hs_short = short_window_size / 2;
    const auto Ha_short = int((float) Hs_short / params.stretch_factor);

    debug_print("Computing mono reference phases...", params.debug);
    std::vector<float> x_sum (x[0].size(), 0.0f);
    for(int ch = 0; ch < (int) x.size(); ++ch)
        for(int n = 0; n < (int) x_sum.size(); ++n)
            x_sum[n] += x[ch][n] / (float) x.size();
    auto X_sum = spectrogram(x_sum, long_window_size, Ha_long);
    const auto phase_mods = phase_propagation(X_sum, params.sample_rate, Ha_long, Hs_long, long_window_size);

    const auto stretch_len = int((float) x[0].size() * params.stretch_factor) + 2000;
    std::vector<std::vector<float>> y;
    for(int ch = 0; ch < (int) x.size(); ++ch)
    {
        debug_print("Processing channel " + std::to_string(ch) + "...", params.debug);
        auto[h_signal, p_signal] = HPSS::hpss(x[ch], params.hpss_params);

        debug_print("Performing time-stretching...", params.debug);
        auto H_full = spectrogram(h_signal, long_window_size, Ha_long);
        auto P_full = spectrogram(p_signal, long_window_size, Ha_long);

        debug_print("\tSeparating magnitude-only PV...", params.debug);
        auto H_long = spectrogram(h_signal, long_window_size, Ha_long);
        const auto h_x_long = reconstruct(H_long, Hs_long, long_window_size, stretch_len);
        auto P_short = spectrogram(p_signal, short_window_size, Ha_short);
        const auto p_x_short = reconstruct(P_short, Hs_short, short_window_size, stretch_len);

        debug_print("\tApplying reference phases...", params.debug);
        for(int m = 0; m < (int) H_full.size(); ++m)
        {
            for(int k = 0; k < (int) H_full[m].size(); ++k)
            {
                H_full[m][k] = std::polar(std::abs(H_full[m][k]), phase_mods[m][k]);
                P_full[m][k] = std::polar(std::abs(P_full[m][k]), phase_mods[m][k]);
            }
        }

        debug_print("\tReconstructing references...", params.debug);
        const auto h_v = reconstruct(H_full, Hs_long, long_window_size, stretch_len);
        const auto p_v = reconstruct(P_full, Hs_long, long_window_size, stretch_len);

        debug_print("\tPerforming magnitude correction...", params.debug);
        auto H_v_long = spectrogram(h_v, long_window_size, Ha_long);
        auto P_v_short = spectrogram(p_v, short_window_size, Ha_short);

        auto H_w_long = spectrogram(h_x_long, long_window_size, Ha_long);
        auto P_w_short = spectrogram(p_x_short, short_window_size, Ha_short);

        for(int m = 0; m < (int) H_w_long.size(); ++m)
            for(int k = 0; k < (int) H_w_long[m].size(); ++k)
                H_v_long[m][k] = std::polar(std::abs(H_w_long[m][k]), std::arg(H_v_long[m][k]));

        const auto M = (int) std::min(P_v_short.size(), P_w_short.size());
        for(int m = 0; m < M; ++m)
            for(int k = 0; k < (int) P_w_short[m].size(); ++k)
                P_v_short[m][k] = std::polar(std::abs(P_w_short[m][k]), std::arg(P_v_short[m][k]));

        debug_print("\tReconstructing final signal...", params.debug);
        auto h_y = reconstruct(H_v_long, Ha_long, long_window_size, stretch_len);
        auto p_y = reconstruct(P_v_short, Ha_short, short_window_size, stretch_len);

        debug_print("\tNormalizing and combining signals...", params.debug);
        normalize_vec(h_signal, h_y);
        normalize_vec(p_signal, p_y);

        y.push_back(std::vector<float> (h_y.size(), 0.0f));
        for(int i = 0; i < (int) h_y.size(); ++i)
            y[ch][i] = h_y[i] + p_y[i];

        normalize_vec(x[ch], y[ch]);
    }

    return y;
}

} // namespace time_stretch
