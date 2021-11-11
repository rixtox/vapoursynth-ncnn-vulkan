/*
  MIT License

  Copyright (c) 2018-2019 HolyWu
  Copyright (c) 2019-2020 NaLan ZeYu

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include <fstream>
#include <algorithm>

#include "filter-common.hpp"
#include "waifu2x-filter.hpp"
#include "gpu.h"
#include "waifu2x.hpp"
#include "vsplugin.hpp"

typedef struct {
    VSNodeRef *node;
    VSVideoInfo vi;
    Waifu2x *waifu2x;
} Waifu2xFilterData;

static int Waifu2xFilter(const VSFrameRef *src, VSFrameRef *dst, Waifu2xFilterData * const VS_RESTRICT d, const VSAPI *vsapi) noexcept {
    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    const int srcStride = vsapi->getStride(src, 0) / static_cast<int>(sizeof(float));
    const int dstStride = vsapi->getStride(dst, 0) / static_cast<int>(sizeof(float));
    auto *             srcR = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0));
    auto *             srcG = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 1));
    auto *             srcB = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 2));
    auto * VS_RESTRICT dstR = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0));
    auto * VS_RESTRICT dstG = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 1));
    auto * VS_RESTRICT dstB = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 2));
    return d->waifu2x->process(srcR, srcG, srcB, dstR, dstG, dstB, width, height, srcStride, dstStride);
}

static void VS_CC Waifu2xFilterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<Waifu2xFilterData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC Waifu2xFilterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<Waifu2xFilterData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef *dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);
        int err = Waifu2xFilter(src, dst, d, vsapi);
        if (err) {
            vsapi->setFilterError("Waifu2x-NCNN-Vulkan: Waifu2x filter error.", frameCtx);
        } else {
            return dst;
        }
    }
    return nullptr;
}

static void VS_CC Waifu2xFilterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<Waifu2xFilterData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d->waifu2x;
    delete d;
    tryDestoryGpuInstance();
}

void VS_CC Waifu2xFilterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    Waifu2xFilterData d{};
    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    int gpuId, ttaMode, noise, scale, model, tileSizeW, tileSizeH, gpuThread, precision;
    std::string paramPath, modelPath;
    char const * err_prompt = nullptr;
    do {
        int err;

        err = tryCreateGpuInstance();
        if (err) {
            err_prompt = "create gpu instance failed";
            break;
        }

        if (!isConstantFormat(&d.vi) || d.vi.format->colorFamily != cmRGB || d.vi.format->sampleType != stFloat || d.vi.format->bitsPerSample != 32) {
            err_prompt = "only constant RGB format and 32 bit float input supported";
            break;
        }

        gpuId = int64ToIntS(vsapi->propGetInt(in, "gpu_id", 0, &err));
        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count()) {
            err_prompt = "invalid 'gpu_id'";
            break;
        }

        ttaMode = int64ToIntS(vsapi->propGetInt(in, "tta_mode", 0, &err));
        if (ttaMode < 0 || ttaMode > 1) {
            err_prompt = "'tta_mode' must be 0 or 1";
            break;
        }

        noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));
        if (noise < -1 || noise > 3) {
            err_prompt = "'noise' must be -1, 0, 1, 2, or 3";
            break;
        }

        scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            scale = 2;
        if (scale != 1 && scale != 2) {
            err_prompt = "'scale' must be 1 or 2";
            break;
        }

        model = int64ToIntS(vsapi->propGetInt(in, "model", 0, &err));
        if (model < 0 || model > 2) {
            err_prompt = "'model' must be 0, 1 or 2";
            break;
        }

        precision = int64ToIntS(vsapi->propGetInt(in, "precision", 0, &err));
        if (err)
            precision = 16;
        if (precision != 16 && precision != 32) {
            err_prompt = "'precision' must be 16 or 32";
            break;
        }

        int customGpuThread = int64ToIntS(vsapi->propGetInt(in, "gpu_thread", 0, &err));
        if (customGpuThread > 0) {
            gpuThread = customGpuThread;
        }
        else {
            gpuThread = int64ToIntS(ncnn::get_gpu_info(gpuId).transfer_queue_count());
        }
        gpuThread = std::min(gpuThread, int64ToIntS(ncnn::get_gpu_info(gpuId).compute_queue_count()));

        int tileSize = int64ToIntS(vsapi->propGetInt(in, "tile_size", 0, &err));
        if (tileSize == 0) {
            double vram = ncnn::get_gpu_device(gpuId)->get_heap_budget(); // in MByte
            double factor = (precision == 32 ? 2 : 1) * (model == 2 ? 1.5 : 1) * gpuThread;
            if (vram / factor > 900)
                tileSize = 360;
            else if (vram / factor > 450)
                tileSize = 240;
            else
                tileSize = 180;
        }
        if (tileSize < 32) {
            err_prompt = "'tile_size' must be greater than or equal to 32";
            break;
        }
        if (tileSize % 4) {
            err_prompt = "'tile_size' must be multiple of 4";
            break;
        }
        tileSizeW = tileSizeH = tileSize;

        int tw = int64ToIntS(vsapi->propGetInt(in, "tile_size_w", 0, &err));
        if (!err) {
            if (tw < 32) {
                err_prompt = "'tile_size_w' must be greater than or equal to 32";
                break;
            }
            if (tw % 4) {
                err_prompt = "'tile_size_w' must be multiple of 4";
                break;
            }
            tileSizeW = tw;
        }

        int th = int64ToIntS(vsapi->propGetInt(in, "tile_size_h", 0, &err));
        if (!err) {
            if (th < 32) {
                err_prompt = "'tile_size_h' must be greater than or equal to 32";
                break;
            }
            if (th % 4) {
                err_prompt = "'tile_size_h' must be multiple of 4";
                break;
            }
            tileSizeH = th;
        }

        if (scale == 1 && noise == -1) {
            err_prompt = "use 'noise=-1' and 'scale=1' at same time is useless";
            break;
        }

        if (scale == 1 && model != 2) {
            err_prompt = "only cunet model support 'scale=1'";
            break;
        }

        // set model path
        const std::string pluginFilePath{ vsapi->getPluginPath(vsapi->getPluginById(VSPLUGIN_IDENTIFIER_STR, core)) };
        const std::string pluginDir = pluginFilePath.substr(0, pluginFilePath.find_last_of('/'));

        std::string modelsDir = pluginDir + "/ncnn-models/Waifu2x/";
        if (model == 0)
            modelsDir += "models-upconv_7_anime_style_art_rgb/";
        else if (model == 1)
            modelsDir += "models-upconv_7_photo/";
        else
            modelsDir += "models-cunet/";

        std::string modelName;
        if (noise == -1)
            modelName = "scale2.0x_model";
        else if (scale == 1)
            modelName = "noise" + std::to_string(noise) + "_model";
        else
            modelName = "noise" + std::to_string(noise) + "_scale2.0x_model";

        paramPath = modelsDir + modelName + ".param";
        modelPath = modelsDir + modelName + ".bin";

        // check model file readable
        std::ifstream pf(paramPath);
        std::ifstream mf(modelPath);
        if (!pf.good() || !mf.good()) {
            err_prompt = "can't open model file";
            break;
        }

        break;
    } while (false);

    if (err_prompt) {
        vsapi->setError(out, (std::string{"Waifu2x-NCNN-Vulkan: "} + err_prompt).c_str());
        vsapi->freeNode(d.node);
        tryDestoryGpuInstance();
        return;
    }

    int prepadding;
    if (model == 2 && scale == 1)
        prepadding = 28;
    else if (model == 2)
        prepadding = 18;
    else
        prepadding = 7;

    d.waifu2x = new Waifu2x(gpuId, gpuThread, ttaMode);
    d.waifu2x->noise = noise;
    d.waifu2x->scale = scale;
    d.waifu2x->tilesize_w = tileSizeW;
    d.waifu2x->tilesize_h = tileSizeH;
    d.waifu2x->prepadding = prepadding;

    d.waifu2x->load(paramPath, modelPath);

    d.vi.width *= scale;
    d.vi.height *= scale;

    auto *data = new Waifu2xFilterData{ d };

    vsapi->createFilter(in, out, "Waifu2x", Waifu2xFilterInit, Waifu2xFilterGetFrame, Waifu2xFilterFree, fmParallel, 0, data, core);
}
