#ifndef WAIFU2X_HPP
#define WAIFU2X_HPP

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class Waifu2x
{
public:
    Waifu2x(int gpuid, int num_threads = 1, bool tta_mode = false);
    ~Waifu2x();

    int load(const std::string& parampath, const std::string& modelpath);

    int process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;

public:
    int noise;
    int scale;
    int tilesize_w;
    int tilesize_h;
    int prepadding;

private:
    ncnn::Net _net;
    ncnn::Pipeline* _preproc;
    ncnn::Pipeline* _postproc;
    bool _tta_mode;
};

#endif
