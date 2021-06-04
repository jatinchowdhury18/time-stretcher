#pragma once

#include <utility>
#include <vector>

namespace HPSS
{

struct HPSS_PARAMS
{
    int perc_kernel = 17; // size of median filter kernel for percussive siganl
    int harm_kernel = 17; // size of median filter kernel for harmonic signal
    float mask_exp = 2.0f; // exponent used for Weiner filter to construct mask
    int fft_size = 4096; // size of window to use for FFT
    int hop_size = 1024; // hop size between FFT windows
    int zero_pad = 2; // zero-padding factor to use for FFT
};

std::pair<std::vector<float>, std::vector<float>> hpss(std::vector<float> x, const HPSS_PARAMS& params);

} // namespace HPSS
