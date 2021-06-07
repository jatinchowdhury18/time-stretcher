#pragma once

#if USING_JUCE

#else
#include <algorithm>
#include <complex> // need to include this before fftw for std::complex compatibility
#include <fftw3.h>
#endif

namespace fft_utils
{

#if USING_JUCE

#else
/** Custom allocator for making vectors compatible with FFTW */
template<typename T>
class fftw_allocator : public std::allocator<T>
{
public:
    template <typename U>
    struct rebind { typedef fftw_allocator<U> other; };
    T* allocate(size_t n) { return (T*) fftwf_malloc(sizeof(T) * n); }
    void deallocate(T* data, std::size_t size) { fftwf_free(data); }
};

using fftw_real_vec = std::vector<float, fftw_allocator<float>>;
using fftw_complex_vec = std::vector<std::complex<float>, fftw_allocator<std::complex<float>>>;

/** Re-interpret std::complex as fftw_complex* (requires including <complex> before fftw3.h) */
inline fftwf_complex* toFFTW (fftw_complex_vec& vec)
{
    return reinterpret_cast<fftwf_complex*> (vec.data());
}

/** Applies a window to the given data (out-of-place) */
static void applyWindow(const std::vector<float>& data, const std::vector<float>& win, std::vector<float>& out)
{
    for(int n = 0; n < (int) win.size(); ++n)
        out[n] = data[n] * win[n];
}

/** Helper struct for performing forward FFTs */
struct ForwardFFT
{
    std::vector<float> x_in;
    fftw_complex_vec Y_out;

    ForwardFFT(int n)
    {
        x_in.resize(n, 0.0f);
        Y_out.resize(n, { 0.0f, 0.0f });
        fft_plan = fftwf_plan_dft_r2c_1d(n, x_in.data(), toFFTW(Y_out), FFTW_ESTIMATE);
    }

    ~ForwardFFT()
    {
        fftwf_destroy_plan(fft_plan);
    }

    void perform()
    {
        fftwf_execute(fft_plan);
    }

private:
    fftwf_plan fft_plan;
};

/** Helper struct for performing inverse FFTs */
struct InverseFFT
{
    fftw_complex_vec X_in;
    std::vector<float> y_out;

    InverseFFT(int n)
    {
        X_in.resize(n, { 0.0f, 0.0f });
        y_out.resize(n, 0.0f);
        fft_plan = fftwf_plan_dft_c2r_1d(n, toFFTW(X_in), y_out.data(), FFTW_ESTIMATE);
        oneOverN = 1.0f / (float) n;
    }

    ~InverseFFT()
    {
        fftwf_destroy_plan(fft_plan);
    }

    void perform()
    {
        fftwf_execute(fft_plan);
        for(int i = 0; i < (int) y_out.size(); ++i)
            y_out[i] *= oneOverN;
    }

private:
    fftwf_plan fft_plan;
    float oneOverN;
};
#endif

/** Helper function for generating Hann window */
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
