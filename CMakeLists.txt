cmake_minimum_required(VERSION 3.16)
project(time-stretch)
set(CMAKE_CXX_STANDARD 17)

include(FindPkgConfig)

add_library(time_stretch STATIC
    src/hpss.cpp
    src/stretch.cpp
)

# Find requirements
if(TIMESTRETCH_USING_JUCE)
    message(STATUS "time-stretch using JUCE")
    target_compile_definitions(time_stretch PUBLIC TIMESTRETCH_USING_JUCE=1)
else()
    message(STATUS "time-stretch using FFTW")
    pkg_check_modules(FFTW REQUIRED fftw3)
    target_include_directories(time_stretch PUBLIC ${FFTW_INCLUDE_DIRS})
    target_link_directories(time_stretch PUBLIC ${FFTW_LIBRARY_DIRS})
    target_link_libraries(time_stretch PUBLIC fftw3f)
endif()

if(BUILD_TESTS)
    message(STATUS "Building tests for time-stretch library")

    pkg_check_modules(SNDFILE REQUIRED sndfile>=1.0.2)

    # test executable for hpss part
    add_executable(hpss src/hpss_test.cpp)
    target_include_directories(hpss PUBLIC ${SNDFILE_INCLUDE_DIRS})
    target_link_directories(hpss PUBLIC ${SNDFILE_LIBRARY_DIRS})
    target_link_libraries(hpss PUBLIC time_stretch ${SNDFILE_LIBRARIES})

    # test executable for time stretching
    add_executable(stretch src/stretch_test.cpp)
    target_include_directories(stretch PUBLIC ${SNDFILE_INCLUDE_DIRS})
    target_link_directories(stretch PUBLIC ${SNDFILE_LIBRARY_DIRS})
    target_link_libraries(stretch PUBLIC time_stretch ${SNDFILE_LIBRARIES})
endif()
