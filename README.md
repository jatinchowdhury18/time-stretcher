# Time Stretcher

C++ time stretching implementation, based on the algorithms
presented in:
- [Audio Time Stretching with an Adaptive Phase Vocoder](http://www.pitchtech.ch/Confs/ICASSP2017/0000716.pdf), Nicolas Juillerat and Beat Hirsbrunner (ICASP 2017)
- [Harmonic/Percussive Separation using Median Filtering](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1078&context=argcon), Derry Fitgerald (DAFx 2010)

TODO before making public:
- test as library with JUCE
  - fix CMake lists...
  - figure out juce::dsp::FFT stuff
  - update README with instructions for linking
- profile performance
- update tests to take arguments

## Dependencies
Building the time-stretching library requires FFTW.
Building the library tests requires libsndfile.

With apt:
```bash
sudo apt-get install fftw3
sudo apt-get install libsndfile
```

Or with HomeBrew:
```bash
brew install fftw3
brew install libsndfile
```

## Building Tests
Building the code requires CMake.
```bash
cmake -Bbuild -DBUILD_TESTS=ON
cmake --build build --config Release
```

`./build/hpss` can be used to test the
harmonic/percussive source separation algorithm.
`./build/stretch` can be used to test the
time-stretching algorithm. Use the `--help`
flag for more information about how to use each test.

## License
The code in this repository is licensed under the BSD 3-clause license.

Enjoy!
