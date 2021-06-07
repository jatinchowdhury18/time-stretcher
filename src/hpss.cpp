#include "hpss.h"
#include "fft_utils.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace HPSS
{

using namespace fft_utils;
using Vec2D = std::vector<std::vector<float>>;

Vec2D median_filter_harm(const std::vector<fftw_complex_vec>& S, int kernel_size)
{
    Vec2D H (S.size(), std::vector<float> (S[0].size(), 0.0f));

    const int pad = (kernel_size - 1) / 2;
    std::vector<float> med_vec (S.size() + 2 * pad, 0.0f);

    for(int h = 0; h < (int) S[0].size(); ++h)
    {
        for(int i = 0; i < (int) S.size(); ++i)
            med_vec[i + pad] = std::abs(S[i][h]);

        for(int i = 0; i < (int) S.size(); ++i)
        {
            std::vector<float> kernel_vec (&med_vec[i], &med_vec[i + kernel_size]);
            std::nth_element(&kernel_vec[0], &kernel_vec[pad], &kernel_vec[kernel_size]);
            H[i][h] = kernel_vec[pad];
        }
    }

    return H;
}

Vec2D median_filter_perc(const std::vector<fftw_complex_vec>& S, int kernel_size)
{
    Vec2D P (S.size(), std::vector<float> (S[0].size(), 0.0f));

    const int pad = (kernel_size - 1) / 2;
    std::vector<float> med_vec (S[0].size() + 2 * pad, 0.0f);

    for(int i = 0; i < (int) S.size(); ++i)
    {
        for(int h = 0; h < (int) S[i].size(); ++h)
            med_vec[h + pad] = std::abs(S[i][h]);

        for(int h = 0; h < (int) S[i].size(); ++h)
        {
            std::vector<float> kernel_vec (&med_vec[h], &med_vec[h + kernel_size]);
            std::nth_element(&kernel_vec[0], &kernel_vec[pad], &kernel_vec[kernel_size]);
            P[i][h] = kernel_vec[pad];
        }
    }

    return P;
}

std::pair<std::vector<float>, std::vector<float>> spec_reconstruct(std::vector<fftw_complex_vec>& H_hat,
                                                                   std::vector<fftw_complex_vec>& P_hat,
                                                                   int n_samples,
                                                                   const HPSS_PARAMS& params)
{
    std::vector<float> h_sig (n_samples, 0.0f);
    std::vector<float> p_sig (n_samples, 0.0f);
    const auto win = hann(params.fft_size, float(params.fft_size / params.hop_size) / 2.0f);

    for(int i = 0; i < H_hat.size(); ++i)
    {
        int start_idx = i * params.hop_size;
        int samples = std::min(params.fft_size, n_samples - start_idx);

        { // do H
            const int n_fft = (int) H_hat[i].size();
            std::vector<float> fft_out (n_fft, 0.0f);
            auto fft_plan = fftwf_plan_dft_c2r_1d(n_fft, toFFTW(H_hat[i]), fft_out.data(), FFTW_ESTIMATE);
            fftwf_execute(fft_plan);
            fftwf_destroy_plan(fft_plan);

            for(int n = 0; n < samples; ++n)
                h_sig[n + start_idx] += win[n] * fft_out[n] / (float)n_fft;
        }

        { // do P
            const int n_fft = (int) P_hat[i].size();
            std::vector<float> fft_out (n_fft, 0.0f);
            auto fft_plan = fftwf_plan_dft_c2r_1d(n_fft, toFFTW(P_hat[i]), fft_out.data(), FFTW_ESTIMATE);
            fftwf_execute(fft_plan);
            fftwf_destroy_plan(fft_plan);

            for(int n = 0; n < samples; ++n)
                p_sig[n + start_idx] += win[n] * fft_out[n] / (float)n_fft;
        }
    }

    return std::make_pair(h_sig, p_sig);
}

std::pair<std::vector<float>, std::vector<float>> hpss(std::vector<float> x, const HPSS_PARAMS& params)
{
    std::cout << "Computing HPSS..." << std::endl;
    std::cout << "\tComputing STFTs..." << std::endl;
    auto S = spectrogram(x, params.fft_size, params.hop_size, params.zero_pad);

    std::cout << "\tSeparating percussive signal..." << std::endl;
    auto P = median_filter_perc(S, params.perc_kernel);

    std::cout << "\tSeparating harmonic signal..." << std::endl;
    auto H = median_filter_harm(S, params.harm_kernel);

    std::cout << "\tApplying filter masks..." << std::endl;
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

    std::cout << "\tComputing time-domain signal..." << std::endl;
    return spec_reconstruct(H_hat, P_hat, (int) x.size(), params);
}

} // namespace HPSS

