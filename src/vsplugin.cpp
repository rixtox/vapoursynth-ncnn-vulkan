#include "vsplugin.hpp"
#include "waifu2x-filter.hpp"
#include "real-esrgan-filter.hpp"
#include "export-frame-filter.hpp"


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc(VSPLUGIN_IDENTIFIER_STR, "ncnn", "VapourSynth NCNN Vulkan Plugin", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Waifu2x",
        "clip:clip;"
        "noise:int:opt;"
        "scale:int:opt;"
        "model:int:opt;"
        "tile_size:int:opt;"
        "gpu_id:int:opt;"
        "tta_mode:int:opt;"
        "gpu_thread:int:opt;"
        "precision:int:opt;"
        "tile_size_w:int:opt;"
        "tile_size_h:int:opt;"
        , Waifu2xFilterCreate, nullptr, plugin);

    registerFunc("RealESRGAN",
        "clip:clip;"
        "scale:int:opt;"
        "model:data:opt;"
        "tile_size:int:opt;"
        "gpu_id:int:opt;"
        "tta_mode:int:opt;"
        "gpu_thread:int:opt;"
        , RealESRGANFilterCreate, nullptr, plugin);

    registerFunc("ExportFrame",
        "dir:data;"
        "prefix:data:opt;"
        "suffix:data:opt;"
        "frame:int:opt;"
        , ExportFrameFilterCreate, nullptr, plugin);
}
