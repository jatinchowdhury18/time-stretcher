cmake_minimum_required(VERSION 3.1)
project(time-stretcher)
set(CMAKE_CXX_STANDARD 17)

include(FindPkgConfig)

# Find requirements
pkg_check_modules(SNDFILE REQUIRED sndfile>=1.0.2)
pkg_check_modules(FFTW REQUIRED fftw3)
include_directories(${SNDFILE_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS})
link_directories(${SNDFILE_LIBRARY_DIRS} ${FFTW_LIBRARY_DIRS})

# test executable for hpss part
add_executable(hpss
    src/hpss_test.cpp
    src/hpss.cpp
)

target_link_libraries(hpss fftw3f ${SNDFILE_LIBRARIES})
