project(vapoursynth-ncnn-vulkan)
cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Release)

set(VapourSynth_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download-ncnn.cmake)

add_subdirectory(src)
