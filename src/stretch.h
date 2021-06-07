#pragma once

#include "hpss.h"

namespace time_stretch
{

struct STRETCH_PARAMS
{
    HPSS::HPSS_PARAMS hpss_params;
    float stretch_factor = 1.0f;
    float sample_rate = 44100.0f;
    float long_window_ms = 100.0f;
    float short_window_ms = 5.0f;
};

std::vector<std::vector<float>> time_stretch(const std::vector<std::vector<float>>& x, const STRETCH_PARAMS& params);

} // namespace time_stretch
