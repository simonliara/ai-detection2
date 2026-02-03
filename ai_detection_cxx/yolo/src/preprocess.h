#pragma once
#include <cstdint>
#include <cuda_runtime.h>

void cuda_preprocess_init(int max_w, int max_h);
void cuda_preprocess_destroy();

void cuda_preprocess(const uint8_t* src, int src_w, int src_h, int src_stride,
                     float* dst_chw, int dst_w, int dst_h,
                     cudaStream_t stream);