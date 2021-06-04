#include "hpss.h"
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

    std::vector<std::vector<float>> h_signal;
    std::vector<std::vector<float>> p_signal;
    std::vector<std::vector<float>> sum_signal;
    for(int ch = 0; ch < (int) ref_signal.size(); ++ch)
    {
        auto[h_ch, p_ch] = HPSS::hpss(ref_signal[ch], HPSS::HPSS_PARAMS());
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
