# Time Stretcher

C++ time stretching implementation, based on the algorithms
presented in:
- [Audio Time Stretching with an Adaptive Phase Vocoder](http://www.pitchtech.ch/Confs/ICASSP2017/0000716.pdf), Nicolas Juillerat, Beat Hirsbrunner
- [Harmonic/Percussive Separation using Median Filtering](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1078&context=argcon)

## Dependencies
Building the code in this repository requires FFTW and libsndfile.

Linux:
```bash
sudo apt-get install fftw3
sudo apt-get install libsndfile
```

MacOS:
```bash
brew install fftw3
brew install libsndfile
```

## Building
Building the code requires CMake.
```bash
cmake -Bbuild
cmake --build build --config Release
```

`./build/hpss` can be used to test the
harmonic/percussive source separation algorithm.
`./build/stretch` can be used to test the
time-stretching algorithm.

## License
