#include "stretch.h"
#include "WavIO.hpp"

int main()
{
    constexpr int fs = 44100;
    constexpr int START_SECONDS = 125;
    constexpr int NUM_SECONDS = 10;
    std::string TEST_FILE = "/Users/jachowdhury/Downloads/Tennyson - Old Singles/Tennyson - Old Singles - 01 All Yours.wav";

    SF_INFO sf_info;
    auto wav_signal = WavIO::load_file(TEST_FILE.c_str(), sf_info);

    // trim signal
    std::vector<std::vector<float>> ref_signal;
    {
        int start_sample = fs * START_SECONDS;
        int end_sample = start_sample + fs * NUM_SECONDS;
        for(int ch = 0; ch < (int) wav_signal.size(); ++ch)
            ref_signal.push_back(std::vector<float> (&wav_signal[ch][start_sample], &wav_signal[ch][end_sample]));
    }

    time_stretch::STRETCH_PARAMS params;
    params.sample_rate = (float) fs;
    params.stretch_factor = 1.5f;
    params.hpss_params.debug = true;

    auto stretch_signal = time_stretch::time_stretch(ref_signal, params);

    WavIO::write_file("ref.wav", ref_signal, sf_info);
    WavIO::write_file("stretch.wav", stretch_signal, sf_info);

    return 0;
}
