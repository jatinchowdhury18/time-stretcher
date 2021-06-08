#include "hpss.h"
#include "WavIO.hpp"

void help()
{
    std::cout << "Utility to separate harmonic and percussive signals from a .wav file" << std::endl;
    std::cout << "Usage: hpss <wav_file> [<num_seconds> <start_seconds>]" << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc < 2 || argc > 4)
    {
        help();
        return 1;
    }

    if(argc == 2 && std::string(argv[1]) == "--help")
    {
        help();
        return 1;
    }

    std::string test_file = std::string(argv[1]);

    float num_seconds = 10.0f;
    if(argc >= 3)
        num_seconds = (float) std::atof(argv[2]);

    float start_seconds = 0.0f;
    if(argc >= 4)
        start_seconds = (float) std::atof(argv[3]);

    SF_INFO sf_info;
    auto wav_signal = WavIO::load_file(test_file.c_str(), sf_info);
    const float fs = (float) sf_info.samplerate;

    // trim signal
    std::vector<std::vector<float>> ref_signal;
    {
        int start_sample = int(fs * start_seconds);
        int end_sample = start_sample + int(fs * num_seconds);
        for(int ch = 0; ch < (int) wav_signal.size(); ++ch)
            ref_signal.push_back(std::vector<float> (&wav_signal[ch][start_sample], &wav_signal[ch][end_sample]));
    }

    HPSS::HPSS_PARAMS params;
    params.sample_rate = fs;
    params.debug = true;

    std::vector<std::vector<float>> h_signal;
    std::vector<std::vector<float>> p_signal;
    std::vector<std::vector<float>> sum_signal;
    for(int ch = 0; ch < (int) ref_signal.size(); ++ch)
    {
        auto[h_ch, p_ch] = HPSS::hpss(ref_signal[ch], params);
        h_signal.push_back(h_ch);
        p_signal.push_back(p_ch);

        std::vector<float> sum_ch (h_ch.size(), 0.0f);
        for(int i = 0; i < (int) h_ch.size(); ++i)
            sum_ch[i] = h_ch[i] + p_ch[i];

        sum_signal.push_back(sum_ch);
    }

    WavIO::write_file("ref.wav", ref_signal, sf_info);
    WavIO::write_file("harmonic.wav", h_signal, sf_info);
    WavIO::write_file("percussive.wav", p_signal, sf_info);
    WavIO::write_file("sum.wav", sum_signal, sf_info);

    return 0;
}
