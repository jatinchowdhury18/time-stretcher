#include "stretch.h"
#include "fft_utils.h"
#include <cmath>
#include <iostream>

namespace time_stretch
{

using namespace fft_utils;

int next_pow2(int x)
{
    return (int) std::pow(2.0f, std::ceil(std::log2((double) x)));
}

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

std::vector<float> reconstruct(std::vector<fftw_complex_vec>& S, int hop_size, int window_size, int L)
{
    std::vector<float> y (L, 0.0f);
    auto win = fft_utils::hann(window_size, 1.0f);
    int final_idx = 0;
    for(int i = 0; i < (int) S.size(); ++i)
    {
        const auto start_idx = int(i * hop_size);
        const auto n_samples = std::min(window_size, L - start_idx);

        const auto n_fft = (int) S[i].size();
        std::vector<float> fft_out (n_fft, 0.0f);
        auto fft_plan = fftwf_plan_dft_c2r_1d(n_fft, toFFTW(S[i]), fft_out.data(), FFTW_ESTIMATE);
        fftwf_execute(fft_plan);
        fftwf_destroy_plan(fft_plan);

        for(int n = 0; n < n_samples; ++n)
            y[n + start_idx] += win[n] * fft_out[n] / (float) n_fft;
        final_idx = start_idx + n_samples;
    }

    return { &y[0], &y[final_idx] };
}

template<typename T>
static bool abs_compare(T a, T b)
{
    return (std::abs(a) < std::abs(b));
}

std::vector<std::vector<float>> time_stretch(const std::vector<std::vector<float>>& x, const STRETCH_PARAMS& params)
{
    const auto long_window_size = next_pow2(int(params.long_window_ms * 0.001 * params.sample_rate));
    const auto short_window_size = next_pow2(int(params.short_window_ms * 0.001 * params.sample_rate));

    const auto Hs_long = long_window_size / 2;
    const auto Ha_long = int((float) Hs_long / params.stretch_factor);

    const auto Hs_short = short_window_size / 2;
    const auto Ha_short = int((float) Hs_short / params.stretch_factor);

    std::cout << "Computing mono reference phases..." << std::endl;
    std::vector<float> x_sum (x[0].size(), 0.0f);
    for(int ch = 0; ch < (int) x.size(); ++ch)
        for(int n = 0; n < (int) x_sum.size(); ++n)
            x_sum[n] += x[ch][n];
    auto X_sum = spectrogram(x_sum, long_window_size, Ha_long);
    const auto phase_mods = phase_propagation(X_sum, params.sample_rate, Ha_long, Hs_long, long_window_size);

    const auto stretch_len = int((float) x[0].size() * params.stretch_factor) + 2000;
    std::vector<std::vector<float>> y;
    for(int ch = 0; ch < (int) x.size(); ++ch)
    {
        std::cout << "Processing channel " << ch << "..." << std::endl;
        auto[h_signal, p_signal] = HPSS::hpss(x[ch], params.hpss_params);

        std::cout << "Performing time-stretching..." << std::endl;
        auto H_full = spectrogram(h_signal, long_window_size, Ha_long);
        auto P_full = spectrogram(p_signal, long_window_size, Ha_long);

        std::cout << "\tSeparating magnitude-only PV..." << std::endl;
        auto H_long = spectrogram(h_signal, long_window_size, Ha_long);
        const auto h_x_long = reconstruct(H_long, Hs_long, long_window_size, stretch_len);
        auto P_short = spectrogram(p_signal, short_window_size, Ha_short);
        const auto p_x_short = reconstruct(P_short, Hs_short, short_window_size, stretch_len);

        std::cout << "\tApplying reference phases..." << std::endl;
        for(int m = 0; m < (int) H_full.size(); ++m)
        {
            for(int k = 0; k < (int) H_full[m].size(); ++k)
            {
                auto phase_adj = std::exp(std::complex<float> { 0.0f, 2.0f * (float)M_PI * phase_mods[m][k] });
                H_full[m][k] *= phase_adj;
                P_full[m][k] *= phase_adj;
            }
        }

        std::cout << "\tReconstructing references..." << std::endl;
        const auto h_v = reconstruct(H_full, Hs_long, long_window_size, stretch_len);
        const auto p_v = reconstruct(P_full, Hs_long, long_window_size, stretch_len);

        std::cout << "\tPerforming magnitude correction..." << std::endl;
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

        std::cout << "\tReconstructing final signal..." << std::endl;
        auto h_y = reconstruct(H_v_long, Ha_long, long_window_size, stretch_len);
        auto p_y = reconstruct(P_v_short, Ha_short, short_window_size, stretch_len);

        std::cout << "\tNormalizing separated signal..." << std::endl;
        auto h_mag_pre = *std::max_element(h_signal.begin(), h_signal.end(), abs_compare<float>);
        auto p_mag_pre = *std::max_element(p_signal.begin(), p_signal.end(), abs_compare<float>);
        auto h_mag_post = *std::max_element(h_y.begin(), h_y.end(), abs_compare<float>);
        auto p_mag_post = *std::max_element(p_y.begin(), p_y.end(), abs_compare<float>);

        y.push_back(std::vector<float> (h_y.size(), 0.0f));
        for(int i = 0; i < (int) h_y.size(); ++i)
            y[ch][i] = (h_y[i] + p_y[i]) * 0.25f;

        std::cout << *std::max_element(y[ch].begin(), y[ch].end(), abs_compare<float>) << std::endl;
    }

    // @TODO: normalize

    return y;
}

} // namespace time_stretch
