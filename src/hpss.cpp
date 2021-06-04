#include "hpss.h"
#include <iostream>
#include <algorithm>
#include <complex> // need to include this before fftw for std::complex compatibility
#include <fftw3.h>
#include <limits>

namespace HPSS
{

// fftw utils
template<typename T>
class fftw_allocator : public std::allocator<T>
{
public:
    template <typename U>
    struct rebind { typedef fftw_allocator<U> other; };
    T* allocate(size_t n) { return (T*) fftwf_malloc(sizeof(T) * n); }
    void deallocate(T* data, std::size_t size) { fftwf_free(data); }
};

using fftw_real_vec = std::vector<float, fftw_allocator<double>>;
using fftw_complex_vec = std::vector<std::complex<float>, fftw_allocator<std::complex<float>>>;

fftwf_complex* toFFTW (fftw_complex_vec& vec)
{
    return reinterpret_cast<fftwf_complex*> (vec.data());
}

std::vector<fftw_complex_vec> spectrogram(std::vector<float> x, const HPSS_PARAMS& params)
{
    const auto n_fft = params.fft_size * params.zero_pad;

    std::vector<fftw_complex_vec> S;
    for(int i = 0; i < (int) x.size(); i += params.hop_size)
    {
        std::vector<float> x_pad (n_fft, 0.0f);
        std::copy(&x[i], &x[i + params.fft_size], x_pad.data());

        fftw_complex_vec spectral_frame (n_fft);

        auto fft_plan = fftwf_plan_dft_r2c_1d(n_fft, x_pad.data(), toFFTW(spectral_frame), FFTW_ESTIMATE);
        fftwf_execute(fft_plan);
        fftwf_destroy_plan(fft_plan);
        S.push_back(spectral_frame);
    }

    return S;
}

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

std::vector<float> hann(int N, float normalization)
{
    std::vector<float> win (N, 0.0f);
    for(int i = 0; i < N; ++i)
    {
        win[i] = 0.5f - 0.5f * std::cos(2 * M_PI * (float)i / (float(N - 1)));
        win[i] /= normalization;
    }

    return win;
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
    std::cout << "Computing STFTs..." << std::endl;
    auto S = spectrogram(x, params);

    std::cout << "Separating percussive signal..." << std::endl;
    auto P = median_filter_perc(S, params.perc_kernel);

    std::cout << "Separating harmonic signal..." << std::endl;
    auto H = median_filter_harm(S, params.harm_kernel);

    std::cout << "Applying filter masks..." << std::endl;
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

    std::cout << "Computing time-domain signal..." << std::endl;
    return spec_reconstruct(H_hat, P_hat, (int) x.size(), params);
}

} // namespace HPSS

