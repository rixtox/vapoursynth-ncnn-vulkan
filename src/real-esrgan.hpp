#ifndef REALESRGAN_HPP
#define REALESRGAN_HPP

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class RealESRGAN
{
public:
    RealESRGAN(int gpuid, int num_threads = 1, bool tta_mode = false);
    ~RealESRGAN();

    int load(const std::string& parampath, const std::string& modelpath);

    int process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;

public:
    int scale;
    int tilesize;
    int prepadding;

private:
    ncnn::Net _net;
    ncnn::Pipeline* _preproc;
    ncnn::Pipeline* _postproc;
    bool _tta_mode;
};

#endif // REALESRGAN_HPP
