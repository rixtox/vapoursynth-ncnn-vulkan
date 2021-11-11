#include "gpu.h"

static ncnn::Mutex instanceLock;
static int instanceCounter = 0;

int tryCreateGpuInstance() {
    ncnn::MutexLockGuard lg(instanceLock);
    if (instanceCounter++ == 0) {
        return ncnn::create_gpu_instance();
    } else {
        return 0;
    }
}

void tryDestoryGpuInstance() {
    ncnn::MutexLockGuard lg(instanceLock);
    if (--instanceCounter == 0) {
        ncnn::destroy_gpu_instance();
    }
}
