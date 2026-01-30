#include "preprocess.h"
#include <algorithm>
#include <cmath>

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

struct AffineMatrix {
    float v[6];
};

__global__ void warp_affine_bilinear_kernel(
    uint8_t* __restrict__ src, int src_stride, int src_w, int src_h,
    float* __restrict__ dst, int dst_w, int dst_h,
    uint8_t const_val, AffineMatrix d2s) 
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    float src_x = d2s.v[0] * dx + d2s.v[1] * dy + d2s.v[2] + 0.5f;
    float src_y = d2s.v[3] * dx + d2s.v[4] * dy + d2s.v[5] + 0.5f;

    float c0 = const_val, c1 = const_val, c2 = const_val;

    if (src_x > -1 && src_x < src_w && src_y > -1 && src_y < src_h) {
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;

        float lx = src_x - x_low;
        float ly = src_y - y_low;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        uint8_t* p1 = (y_low >= 0 && x_low >= 0) ? src + y_low * src_stride + x_low * 3 : nullptr;
        uint8_t* p2 = (y_low >= 0 && x_high < src_w) ? src + y_low * src_stride + x_high * 3 : nullptr;
        uint8_t* p3 = (y_high < src_h && x_low >= 0) ? src + y_high * src_stride + x_low * 3 : nullptr;
        uint8_t* p4 = (y_high < src_h && x_high < src_w) ? src + y_high * src_stride + x_high * 3 : nullptr;

        auto get_val = [&](uint8_t* p, int ch) -> uint8_t { return p ? p[ch] : const_val; };

        c0 = w1 * get_val(p1, 0) + w2 * get_val(p2, 0) + w3 * get_val(p3, 0) + w4 * get_val(p4, 0);
        c1 = w1 * get_val(p1, 1) + w2 * get_val(p2, 1) + w3 * get_val(p3, 1) + w4 * get_val(p4, 1);
        c2 = w1 * get_val(p1, 2) + w2 * get_val(p2, 2) + w3 * get_val(p3, 2) + w4 * get_val(p4, 2);
    }

    float t = c2; c2 = c0; c0 = t;

    int area = dst_w * dst_h;
    int offset = dy * dst_w + dx;
    
    dst[offset] = c0 / 255.0f;
    dst[offset + area] = c1 / 255.0f;
    dst[offset + area * 2] = c2 / 255.0f;
}

void cuda_preprocess_init(int max_image_size) {
    if (!img_buffer_host) 
        cudaMallocHost((void**)&img_buffer_host, max_image_size * 3);
    if (!img_buffer_device) 
        cudaMalloc((void**)&img_buffer_device, max_image_size * 3);
}

void cuda_preprocess_destroy() {
    if (img_buffer_device) { cudaFree(img_buffer_device); img_buffer_device = nullptr; }
    if (img_buffer_host) { cudaFreeHost(img_buffer_host); img_buffer_host = nullptr; }
}

void cuda_preprocess(uint8_t* src, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height,
                     cudaStream_t stream) 
{
    int img_size = src_width * src_height * 3;
    
    memcpy(img_buffer_host, src, img_size);
    cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream);

    AffineMatrix d2s;
    float scale = std::min((float)dst_height / src_height, (float)dst_width / src_width);
    
    float tx = -scale * src_width * 0.5f + dst_width * 0.5f;
    float ty = -scale * src_height * 0.5f + dst_height * 0.5f;

    double inv_scale = 1.0 / scale;
    d2s.v[0] = inv_scale;
    d2s.v[1] = 0;
    d2s.v[2] = -tx * inv_scale;
    d2s.v[3] = 0;
    d2s.v[4] = inv_scale;
    d2s.v[5] = -ty * inv_scale;

    dim3 threads(32, 32);
    dim3 blocks((dst_width + threads.x - 1) / threads.x, 
                (dst_height + threads.y - 1) / threads.y);

    warp_affine_bilinear_kernel<<<blocks, threads, 0, stream>>>(
        img_buffer_device, src_width * 3, src_width, src_height,
        dst, dst_width, dst_height,
        114,
        d2s
    );
}