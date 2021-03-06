#include <vector>
#include <algorithm>

#include "real-esrgan.hpp"

static const uint32_t realesrgan_preproc_spv_data[] = {
    #include "realesrgan_preproc.spv.hex.h"
};
static const uint32_t realesrgan_preproc_fp16s_spv_data[] = {
    #include "realesrgan_preproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_int8s_spv_data[] = {
    #include "realesrgan_preproc_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_spv_data[] = {
    #include "realesrgan_postproc.spv.hex.h"
};
static const uint32_t realesrgan_postproc_fp16s_spv_data[] = {
    #include "realesrgan_postproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_int8s_spv_data[] = {
    #include "realesrgan_postproc_int8s.spv.hex.h"
};

static const uint32_t realesrgan_preproc_tta_spv_data[] = {
    #include "realesrgan_preproc_tta.spv.hex.h"
};
static const uint32_t realesrgan_preproc_tta_fp16s_spv_data[] = {
    #include "realesrgan_preproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_tta_int8s_spv_data[] = {
    #include "realesrgan_preproc_tta_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_spv_data[] = {
    #include "realesrgan_postproc_tta.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_fp16s_spv_data[] = {
    #include "realesrgan_postproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_int8s_spv_data[] = {
    #include "realesrgan_postproc_tta_int8s.spv.hex.h"
};

RealESRGAN::RealESRGAN(int gpuid, int num_threads, bool tta_mode)
{
    _net.opt.use_vulkan_compute = true;
    _net.opt.use_fp16_packed = true;
    _net.opt.use_fp16_storage = true;
    _net.opt.use_fp16_arithmetic = false;
    _net.opt.use_int8_storage = false;
    _net.opt.use_int8_arithmetic = false;
    _net.opt.num_threads = num_threads;

    _tta_mode = tta_mode;
    _preproc = nullptr;
    _preproc = nullptr;

    _net.set_vulkan_device(gpuid);
}

RealESRGAN::~RealESRGAN()
{
    // cleanup preprocess and postprocess pipeline
    if (_preproc) delete _preproc;
    if (_postproc) delete _postproc;
}

int RealESRGAN::load(const std::string& parampath, const std::string& modelpath)
{
    _net.load_param(parampath.c_str());
    _net.load_model(modelpath.c_str());

    // initialize preprocess and postprocess pipeline
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        _preproc = new ncnn::Pipeline(_net.vulkan_device());
        _preproc->set_optimal_local_size_xyz(32, 32, 3);

        _postproc = new ncnn::Pipeline(_net.vulkan_device());
        _postproc->set_optimal_local_size_xyz(32, 32, 3);

        if (_tta_mode)
        {
            if (_net.opt.use_fp16_storage && _net.opt.use_int8_storage)
                _preproc->create(realesrgan_preproc_tta_int8s_spv_data, sizeof(realesrgan_preproc_tta_int8s_spv_data), specializations);
            else if (_net.opt.use_fp16_storage)
                _preproc->create(realesrgan_preproc_tta_fp16s_spv_data, sizeof(realesrgan_preproc_tta_fp16s_spv_data), specializations);
            else
                _preproc->create(realesrgan_preproc_tta_spv_data, sizeof(realesrgan_preproc_tta_spv_data), specializations);

            if (_net.opt.use_fp16_storage && _net.opt.use_int8_storage)
                _postproc->create(realesrgan_postproc_tta_int8s_spv_data, sizeof(realesrgan_postproc_tta_int8s_spv_data), specializations);
            else if (_net.opt.use_fp16_storage)
                _postproc->create(realesrgan_postproc_tta_fp16s_spv_data, sizeof(realesrgan_postproc_tta_fp16s_spv_data), specializations);
            else
                _postproc->create(realesrgan_postproc_tta_spv_data, sizeof(realesrgan_postproc_tta_spv_data), specializations);
        }
        else
        {
            if (_net.opt.use_fp16_storage && _net.opt.use_int8_storage)
                _preproc->create(realesrgan_preproc_int8s_spv_data, sizeof(realesrgan_preproc_int8s_spv_data), specializations);
            else if (_net.opt.use_fp16_storage)
                _preproc->create(realesrgan_preproc_fp16s_spv_data, sizeof(realesrgan_preproc_fp16s_spv_data), specializations);
            else
                _preproc->create(realesrgan_preproc_spv_data, sizeof(realesrgan_preproc_spv_data), specializations);

            if (_net.opt.use_fp16_storage && _net.opt.use_int8_storage)
                _postproc->create(realesrgan_postproc_int8s_spv_data, sizeof(realesrgan_postproc_int8s_spv_data), specializations);
            else if (_net.opt.use_fp16_storage)
                _postproc->create(realesrgan_postproc_fp16s_spv_data, sizeof(realesrgan_postproc_fp16s_spv_data), specializations);
            else
                _postproc->create(realesrgan_postproc_spv_data, sizeof(realesrgan_postproc_spv_data), specializations);
        }
    }

    return 0;
}

int RealESRGAN::process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int w, int h, int src_stride, int dst_stride) const
{
    const int channels = 3;
    const int elempack = 1;

    const int TILE_SIZE_W = tilesize;
    const int TILE_SIZE_H = tilesize;

    ncnn::VkAllocator* blob_vkallocator = _net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = _net.vulkan_device()->acquire_staging_allocator();

    ncnn::Option opt = _net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // each tile 100x100
    const int xtiles = (w + TILE_SIZE_W - 1) / TILE_SIZE_W;
    const int ytiles = (h + TILE_SIZE_H - 1) / TILE_SIZE_H;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_H, h) - yi * TILE_SIZE_H;

        int in_tile_y0 = std::max(yi * TILE_SIZE_H - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_H + prepadding, h);
        const int in_tile_w = w;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in;
        in.create(in_tile_w, in_tile_h, channels, sizeof(float));

        float *in_tile_r = in.channel(0);
        float *in_tile_g = in.channel(1);
        float *in_tile_b = in.channel(2);
        const float *sr = srcpR + in_tile_y0 * src_stride;
        const float *sg = srcpG + in_tile_y0 * src_stride;
        const float *sb = srcpB + in_tile_y0 * src_stride;
        for (int y = 0; y < in_tile_h; y++)
        {
            for (int x = 0; x < in_tile_w; x++)
            {
                in_tile_r[in_tile_w * y + x] = sr[src_stride * y + x] * 255.f;
                in_tile_g[in_tile_w * y + x] = sg[src_stride * y + x] * 255.f;
                in_tile_b[in_tile_w * y + x] = sb[src_stride * y + x] * 255.f;
            }
        }

        ncnn::VkCompute cmd(_net.vulkan_device());

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_H, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_H, h);

        ncnn::VkMat out_gpu;
        out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, sizeof(float), blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_W, w) - xi * TILE_SIZE_W;

            if (_tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat dummy_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_W - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_W, w) + prepadding;
                    int tile_y0 = yi * TILE_SIZE_H - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_H, h) + prepadding;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[channels].create(tile_x1 - tile_x0, tile_y1 - tile_y0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, channels, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, channels, in_out_tile_elemsize, 1, blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = dummy_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_W;
                    constants[9].i = std::min(yi * TILE_SIZE_H, prepadding);
                    constants[10].i = channels;
                    constants[11].i = dummy_alpha_tile_gpu.w;
                    constants[12].i = dummy_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(_preproc, bindings, constants, dispatcher);
                }

                // realesrgan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {
                    ncnn::Extractor ex = _net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("data", in_tile_gpu[ti]);

                    ex.extract("output", out_tile_gpu[ti], cmd);
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = dummy_alpha_tile_gpu;
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_W * scale;
                    constants[7].i = std::min(TILE_SIZE_W * scale, out_gpu.w - xi * TILE_SIZE_W * scale);
                    constants[8].i = prepadding * scale;
                    constants[9].i = prepadding * scale;
                    constants[10].i = channels;
                    constants[11].i = dummy_alpha_tile_gpu.w;
                    constants[12].i = dummy_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_W * scale, out_gpu.w - xi * TILE_SIZE_W * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(_postproc, bindings, constants, dispatcher);
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat dummy_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_W - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_W, w) + prepadding;
                    int tile_y0 = yi * TILE_SIZE_H - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_H, h) + prepadding;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = dummy_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_W;
                    constants[9].i = std::min(yi * TILE_SIZE_H, prepadding);
                    constants[10].i = channels;
                    constants[11].i = dummy_alpha_tile_gpu.w;
                    constants[12].i = dummy_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(_preproc, bindings, constants, dispatcher);
                }

                // realesrgan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = _net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("data", in_tile_gpu);

                    ex.extract("output", out_tile_gpu, cmd);
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = dummy_alpha_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_W * scale;
                    constants[7].i = std::min(TILE_SIZE_W * scale, out_gpu.w - xi * TILE_SIZE_W * scale);
                    constants[8].i = prepadding * scale;
                    constants[9].i = prepadding * scale;
                    constants[10].i = channels;
                    constants[11].i = dummy_alpha_tile_gpu.w;
                    constants[12].i = dummy_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_W * scale, out_gpu.w - xi * TILE_SIZE_W * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // download
        {
            ncnn::Mat out;

            cmd.record_clone(out_gpu, out, opt);
            cmd.submit_and_wait();

            const float* out_tile_r = out.channel(0);
            const float* out_tile_g = out.channel(1);
            const float* out_tile_b = out.channel(2);
            float* dr = dstpR + yi * TILE_SIZE_H * scale * dst_stride;
            float* dg = dstpG + yi * TILE_SIZE_H * scale * dst_stride;
            float* db = dstpB + yi * TILE_SIZE_H * scale * dst_stride;
            for (int y = 0; y < out.h; y++)
            {
                for (int x = 0; x < out.w; x++)
                {
                    dr[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_r[out.w * y + x] / 255.f));
                    dg[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_g[out.w * y + x] / 255.f));
                    db[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_b[out.w * y + x] / 255.f));
                }
            }
        }
    }

    _net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    _net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
