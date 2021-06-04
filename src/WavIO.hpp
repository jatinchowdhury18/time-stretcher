#pragma once

#include <sndfile.h>
#include <string>
#include <iostream>
#include <vector>

namespace WavIO
{

using Vec2d = std::vector<std::vector<float>>;
using SND_PTR = std::unique_ptr<SNDFILE, decltype(&sf_close)>;

Vec2d load_file (const char* file, SF_INFO& sf_info)
{
    std::cout << "Loading file: " << file << std::endl;

    SND_PTR wavFile { sf_open(file, SFM_READ, &sf_info), &sf_close };

    if (sf_info.frames == 0)
    {
        std::cout << "File could not be opened!" << std::endl;
        exit (1);
    }

    std::vector<float> readInterleaved(sf_info.channels * sf_info.frames, 0.0);
    sf_readf_float(wavFile.get(), readInterleaved.data(), sf_info.frames);

    Vec2d audio (sf_info.channels, std::vector<float> (sf_info.frames, 0.0));

    // de-interleave channels
    for (int i = 0; i < sf_info.frames; ++i)
    {
        int interleavedPtr = i * sf_info.channels;
        for(size_t ch = 0; ch < sf_info.channels; ++ch)
            audio[ch][i] = readInterleaved[interleavedPtr + ch];
    }

    return audio;
}

void write_file (const char* file, const Vec2d& audio, SF_INFO& sf_info)
{
    std::cout << "Writing to file: " << file << std::endl;

    const auto channels = (int) audio.size();
    const auto frames = (sf_count_t) audio[0].size();
    sf_info.frames = frames;

    SND_PTR wavFile { sf_open(file, SFM_WRITE, &sf_info), &sf_close };
    std::vector<float> writeInterleaved(channels * frames, 0.0);

    // de-interleave channels
    for (int i = 0; i < frames; ++i)
    {
        int interleavedPtr = i * channels;
        for(int ch = 0; ch < channels; ++ch)
            writeInterleaved[interleavedPtr + ch] = audio[ch][i];
    }

    sf_writef_float(wavFile.get(), writeInterleaved.data(), frames);
}

} // namespace WavIO
