#pragma once

#include "hpss.h"

/**
 * Algorithm for time-stretching audio without altering
 * the pitch. Based on the algorithm proposed by Nicolas
 * Juillerat and Beat Hirsbrunner at ICASP 2017:
 * http://www.pitchtech.ch/Confs/ICASSP2017/0000716.pdf
 * 
 * The basic idea is to separate harmonic and percussive
 * parts of the signal and stretch them using a phase vocoder
 * with different window sizes for the different parts of
 * the signal. However, a single phase correction is used
 * across all parts and all channels to preserve phase
 * coherence.
 */ 
namespace time_stretch
{

struct STRETCH_PARAMS
{
    HPSS::HPSS_PARAMS hpss_params;  // parameters for harmonic/percussive source-separation
    float stretch_factor = 1.0f;    // the time-stretching factor used by the stretcher
    float sample_rate = 44100.0f;   // the sample-rate of the incomind audio signal
    float long_window_ms = 100.0f;  // the (long) window length to use for the harmonic signal
    float short_window_ms = 4.0f;   // the (short) window length to use for the percussive signal
    bool debug = false;             // enable print debug statements for the algorithm
};

/** Performs time-stretching on a multi-channel audio signal */
std::vector<std::vector<float>> time_stretch(const std::vector<std::vector<float>>& x, STRETCH_PARAMS& params);

} // namespace time_stretch
