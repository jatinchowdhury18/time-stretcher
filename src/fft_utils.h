#pragma once

#include <algorithm>
#include <complex> // need to include this before fftw for std::complex compatibility
#include <fftw3.h>

namespace fft_utils
{

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

inline fftwf_complex* toFFTW (fftw_complex_vec& vec)
{
    return reinterpret_cast<fftwf_complex*> (vec.data());
}

inline std::vector<fftw_complex_vec> spectrogram(const std::vector<float>& x, int fft_size, int hop_size, int zero_pad = 1)
{
    const auto n_fft = fft_size * zero_pad;

    std::vector<fftw_complex_vec> S;
    for(int i = 0; i + fft_size < (int) x.size(); i += hop_size)
    {
        std::vector<float> x_pad (n_fft, 0.0f);
        std::copy(&x[i], &x[i + fft_size], x_pad.data());

        fftw_complex_vec spectral_frame (n_fft);

        auto fft_plan = fftwf_plan_dft_r2c_1d(n_fft, x_pad.data(), toFFTW(spectral_frame), FFTW_ESTIMATE);
        fftwf_execute(fft_plan);
        fftwf_destroy_plan(fft_plan);
        S.push_back(spectral_frame);
    }

    return S;
}

inline std::vector<float> hann(int N, float normalization = 1.0f)
{
    std::vector<float> win (N, 0.0f);
    for(int i = 0; i < N; ++i)
    {
        win[i] = 0.5f - 0.5f * std::cos(2 * M_PI * (float)i / (float(N - 1)));
        win[i] /= normalization;
    }

    return win;
}

} // namespace fft_utils
