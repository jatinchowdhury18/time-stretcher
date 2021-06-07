cmake_minimum_required(VERSION 3.1)
project(time-stretch)
set(CMAKE_CXX_STANDARD 17)

include(FindPkgConfig)

# Find requirements
pkg_check_modules(SNDFILE REQUIRED sndfile>=1.0.2)
pkg_check_modules(FFTW REQUIRED fftw3)
include_directories(${SNDFILE_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS})
link_directories(${SNDFILE_LIBRARY_DIRS} ${FFTW_LIBRARY_DIRS})

add_library(time_stretch STATIC
    src/hpss.cpp
    src/stretch.cpp
)

target_link_libraries(time_stretch fftw3f)

if(BUILD_TESTS)
    # test executable for hpss part
    add_executable(hpss src/hpss_test.cpp)
    target_link_libraries(hpss time_stretch ${SNDFILE_LIBRARIES})

    # test executable for time stretching
    add_executable(stretch src/stretch_test.cpp)
    target_link_libraries(stretch time_stretch ${SNDFILE_LIBRARIES})
endif()
