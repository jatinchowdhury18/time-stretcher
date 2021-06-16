#include "stretch.h"
#include "WavIO.hpp"

void help()
{
    std::cout << "Utility to time-stretch a .wav file" << std::endl;
    std::cout << "Usage: hpss <wav_file> <stretch_factor> [<num_seconds> <start_seconds>]" << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc < 3 || argc > 5)
    {
        help();
        return 1;
    }

    std::string test_file = std::string(argv[1]);
    const auto stretch_factor = (float) std::atof(argv[2]);

    float num_seconds = 10.0f;
    if(argc >= 4)
        num_seconds = (float) std::atof(argv[3]);

    float start_seconds = 0.0f;
    if(argc >= 5)
        start_seconds = (float) std::atof(argv[4]);

    SF_INFO sf_info;
    auto wav_signal = WavIO::load_file(test_file.c_str(), sf_info);
    const float fs = (float) sf_info.samplerate;

    // trim signal
    std::vector<std::vector<float>> ref_signal;
    {
        int start_sample = int(fs * start_seconds);
        int end_sample = std::min(start_sample + int(fs * num_seconds), (int) sf_info.frames);
        for(int ch = 0; ch < (int) wav_signal.size(); ++ch)
            ref_signal.push_back(std::vector<float> (&wav_signal[ch][start_sample], &wav_signal[ch][end_sample]));
    }

    time_stretch::STRETCH_PARAMS params;
    params.sample_rate = fs;
    params.stretch_factor = stretch_factor;
    params.debug = true;
    params.hpss_params.debug = true;

    auto stretch_signal = time_stretch::time_stretch(ref_signal, params);

    WavIO::write_file("ref.wav", ref_signal, sf_info);
    WavIO::write_file("stretch.wav", stretch_signal, sf_info);

    return 0;
}
