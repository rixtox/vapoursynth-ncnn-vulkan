find_path(VapourSynthInclude vapoursynth/VapourSynth.h
    HINTS $ENV{ProgramFiles}/VapourSynth/sdk/include
    REQUIRED
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_VapourSynthLib lib64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(_VapourSynthLib lib32)
endif()

find_library(VapourSynthLib vapoursynth vsscript
    HINTS $ENV{ProgramFiles}/VapourSynth/sdk/${_VapourSynthLib}
    REQUIRED
)

find_library(VSScriptLib vapoursynth vsscript
    HINTS $ENV{ProgramFiles}/VapourSynth/sdk/${_VapourSynthLib}
    REQUIRED
)

add_library(VapourSynth INTERFACE)
target_link_libraries(VapourSynth INTERFACE ${VapourSynthLib} ${VSScriptLib})
target_include_directories(VapourSynth INTERFACE ${VapourSynthInclude})
