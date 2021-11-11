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
#include "real-esrgan-filter.hpp"
#include "gpu.h"
#include "real-esrgan.hpp"
#include "vsplugin.hpp"

typedef struct {
    VSNodeRef *node;
    VSVideoInfo vi;
    RealESRGAN *real_esrgan;
} RealESRGANFilterData;

static int RealESRGANFilter(const VSFrameRef *src, VSFrameRef *dst, RealESRGANFilterData * const VS_RESTRICT d, const VSAPI *vsapi) noexcept {
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
    return d->real_esrgan->process(srcR, srcG, srcB, dstR, dstG, dstB, width, height, srcStride, dstStride);
}

static void VS_CC RealESRGANFilterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<RealESRGANFilterData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC RealESRGANFilterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<RealESRGANFilterData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef *dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);
        int err = RealESRGANFilter(src, dst, d, vsapi);
        if (err) {
            vsapi->setFilterError("RealESRGAN-NCNN-Vulkan: RealESRGAN filter error.", frameCtx);
        } else {
            return dst;
        }
    }
    return nullptr;
}

static void VS_CC RealESRGANFilterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<RealESRGANFilterData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d->real_esrgan;
    delete d;
    tryDestoryGpuInstance();
}

void VS_CC RealESRGANFilterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    RealESRGANFilterData d{};
    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    int gpuId, ttaMode, scale, tileSize, gpuThread;
    std::string modelName, paramPath, modelPath;
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

        scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            scale = 4;
        if (scale != 4) {
            err_prompt = "'scale' must be 4";
            break;
        }

        modelName = std::string(vsapi->propGetData(in, "model", 0, &err));
        if (err)
            modelName = "realesrgan-x4plus";

        int customGpuThread = int64ToIntS(vsapi->propGetInt(in, "gpu_thread", 0, &err));
        if (customGpuThread > 0) {
            gpuThread = customGpuThread;
        }
        else {
            gpuThread = int64ToIntS(ncnn::get_gpu_info(gpuId).transfer_queue_count());
        }
        gpuThread = std::min(gpuThread, int64ToIntS(ncnn::get_gpu_info(gpuId).compute_queue_count()));

        tileSize = int64ToIntS(vsapi->propGetInt(in, "tile_size", 0, &err));
        if (err || tileSize == 0) {
            double heap_budget = ncnn::get_gpu_device(gpuId)->get_heap_budget(); // in MByte
            if (heap_budget > 1900)
                tileSize = 200;
            else if (heap_budget > 550)
                tileSize = 100;
            else if (heap_budget > 190)
                tileSize = 64;
            else
                tileSize = 32;
        }
        if (tileSize < 32) {
            err_prompt = "'tile_size' must be greater than or equal to 32";
            break;
        }
        if (tileSize % 4) {
            err_prompt = "'tile_size' must be multiple of 4";
            break;
        }

        // set model path
        const std::string pluginFilePath{ vsapi->getPluginPath(vsapi->getPluginById(VSPLUGIN_IDENTIFIER_STR, core)) };
        const std::string pluginDir = pluginFilePath.substr(0, pluginFilePath.find_last_of('/'));

        std::string modelsDir = pluginDir + "/ncnn-models/Real-ESRGAN/";

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
        vsapi->setError(out, (std::string{"RealESRGAN-NCNN-Vulkan: "} + err_prompt).c_str());
        vsapi->freeNode(d.node);
        tryDestoryGpuInstance();
        return;
    }

    int prepadding = 10;

    d.real_esrgan = new RealESRGAN(gpuId, gpuThread, ttaMode);
    d.real_esrgan->scale = scale;
    d.real_esrgan->tilesize = tileSize;
    d.real_esrgan->prepadding = prepadding;

    d.real_esrgan->load(paramPath, modelPath);

    d.vi.width *= scale;
    d.vi.height *= scale;

    auto *data = new RealESRGANFilterData{ d };

    vsapi->createFilter(in, out, "RealESRGAN", RealESRGANFilterInit, RealESRGANFilterGetFrame, RealESRGANFilterFree, fmParallel, 0, data, core);
}
