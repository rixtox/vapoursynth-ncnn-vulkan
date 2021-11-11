cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(${PROJECT_SOURCE_DIR}/cmake/compile-shader.cmake)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED CONFIG)
find_package(VapourSynth REQUIRED)

find_program(GLSLANGVALIDATOR_EXECUTABLE
    NAMES glslangValidator
    PATHS $ENV{VULKAN_SDK}/bin
    REQUIRED
    NO_CMAKE_FIND_ROOT_PATH)

file(GLOB SOURCE_FILES *.cpp *.hpp)
file(GLOB SHADER_FILES shaders/*.comp)

# generate-spirv target
set(SHADER_SPV_HEX_FILES)
compile_shader(SHADER_SPV_HEX_FILES ${SHADER_FILES})
add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_library(vsnvk SHARED ${SOURCE_FILES})
target_link_libraries(vsnvk PRIVATE Threads::Threads OpenMP::OpenMP_CXX ncnn VapourSynth)
target_include_directories(vsnvk PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(vsnvk generate-spirv)
