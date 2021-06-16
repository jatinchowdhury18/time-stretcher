#pragma once

#include <utility>
#include <vector>

/**
 * Algorithm for harmonic/percussive source-separation
 * using median filtering. Based on the algorithm
 * proposed by Derry Fitzgerald at DAFx 2010:
 * https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1078&context=argcon
 */ 
namespace HPSS
{

struct HPSS_PARAMS
{
    int perc_kernel = 17;           // size of median filter kernel for percussive siganl
    int harm_kernel = 17;           // size of median filter kernel for harmonic signal
    float mask_exp = 2.0f;          // exponent used for Weiner filter to construct mask
    float sample_rate = 44100.0f;   // sample rate of the audio being processed
    float window_length_ms = 40.0f; // length of window to use for FFT
    float hop_length_ms = 20.0f;    // length of hop size to use for FFT (should 1x, 0.5x, or 0.25x of the window length)
    int zero_pad = 2;               // zero-padding factor to use for FFT
    bool debug = false;             // enable print debug statements for the algorithm
};

/**
 * Accepts a single vector of audio samples, and returns a pair of vectors
 * with the separate harmonic and percussive signal.
 * 
 * ```
 * std::vector<float> audio;
 * HPSS_PARAMS params;
 * auto [harmonic, percussive] = hpss(audio, params);
 * ```
 */ 
std::pair<std::vector<float>, std::vector<float>> hpss(std::vector<float> x, const HPSS_PARAMS& params);

} // namespace HPSS
