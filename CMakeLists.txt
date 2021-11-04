project(vapoursynth-waifu2x-ncnn-vulkan)
cmake_minimum_required(VERSION 3.12)

set(VapourSynth_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download-ncnn.cmake)

find_package(ncnn REQUIRED CONFIG)
find_package(VapourSynth REQUIRED)

# libvsw2xnvk
add_library(vsw2xnvk SHARED src/vsw2xnvk.cpp src/waifu2x.cpp)
target_link_libraries(vsw2xnvk ncnn VapourSynth)
