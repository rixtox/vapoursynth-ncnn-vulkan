#include <string>
#include <vapoursynth/VSHelper.h>

#include "filter-common.hpp"
#include "export-frame-filter.hpp"

typedef struct {
    std::string dir;
    std::string prefix;
    std::string suffix;
    int frame;
    VSVideoInfo vi;
    VSNodeRef *node;
    const VSAPI *vsapi;
} ExportFrameFilterData;

static void ExportFrameFilterDataFreeAndClear(ExportFrameFilterData **d)
{
    if ((*d)->vsapi != nullptr && (*d)->node != nullptr) {
        (*d)->vsapi->freeNode((*d)->node);
        (*d)->node = nullptr;
    }
    FreeAndClear(d);
}

static void VS_CC ExportFrameFilterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    ExportFrameFilterData *d = static_cast<ExportFrameFilterData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC ExportFrameFilterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    ExportFrameFilterData *d = static_cast<ExportFrameFilterData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        return src;
    }
    return nullptr;
}

static void VS_CC ExportFrameFilterFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    ExportFrameFilterData *d = static_cast<ExportFrameFilterData *>(instanceData);
    ExportFrameFilterDataFreeAndClear(&d);
}

void VS_CC ExportFrameFilterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    int err;
    char const * err_prompt = nullptr;
    ExportFrameFilterData *d = new ExportFrameFilterData {};
    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    d->vsapi = vsapi;

    do {
        vsapi->propGetData(in, "dir", 0, &err);
        if (err) {
            err_prompt = "'dir' must be set";
            break;
        }
    } while (false);

    if (err_prompt) {
        vsapi->setError(out, (std::string{"Export-Frame-Filter: "} + err_prompt).c_str());
        vsapi->freeNode(d->node);
        return;
    }
    
    vsapi->createFilter(in, out, "ExportFrame", ExportFrameFilterInit, ExportFrameFilterGetFrame, ExportFrameFilterFree, fmParallel, 0, d, core);
    d = nullptr;

bail:
    ExportFrameFilterDataFreeAndClear(&d);
}
