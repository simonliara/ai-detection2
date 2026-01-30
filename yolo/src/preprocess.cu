#include "preprocess.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

static uint8_t* g_src_device = nullptr;   // packed BGR on device
static int g_max_w = 0;
static int g_max_h = 0;

struct AffineMatrix { float v[6]; };

static inline void checkCuda(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(e)
                  << " at " << file << ":" << line
                  << " (" << what << ")\n";
        std::abort();
    }
}
#define CHECK_CUDA(x) checkCuda((x), #x, __FILE__, __LINE__)

__global__ void warp_affine_bilinear_kernel(
    const uint8_t* __restrict__ src, int src_stride, int src_w, int src_h,
    float* __restrict__ dst, int dst_w, int dst_h,
    uint8_t const_val, AffineMatrix d2s)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float src_x = d2s.v[0] * dx + d2s.v[1] * dy + d2s.v[2] + 0.5f;
    float src_y = d2s.v[3] * dx + d2s.v[4] * dy + d2s.v[5] + 0.5f;

    float b = const_val, g = const_val, r = const_val;

    if (src_x > -1 && src_x < src_w && src_y > -1 && src_y < src_h) {
        int x_low  = floorf(src_x);
        int y_low  = floorf(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;

        float lx = src_x - x_low;
        float ly = src_y - y_low;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        auto ptr = [&](int x, int y) -> const uint8_t* {
            if (x < 0 || x >= src_w || y < 0 || y >= src_h) return nullptr;
            return src + y * src_stride + x * 3;
        };

        const uint8_t* p1 = ptr(x_low,  y_low);
        const uint8_t* p2 = ptr(x_high, y_low);
        const uint8_t* p3 = ptr(x_low,  y_high);
        const uint8_t* p4 = ptr(x_high, y_high);

        auto getc = [&](const uint8_t* p, int c)->float { return p ? (float)p[c] : (float)const_val; };

        b = w1 * getc(p1,0) + w2 * getc(p2,0) + w3 * getc(p3,0) + w4 * getc(p4,0);
        g = w1 * getc(p1,1) + w2 * getc(p2,1) + w3 * getc(p3,1) + w4 * getc(p4,1);
        r = w1 * getc(p1,2) + w2 * getc(p2,2) + w3 * getc(p3,2) + w4 * getc(p4,2);
    }

    int area   = dst_w * dst_h;
    int offset = dy * dst_w + dx;

    dst[offset + 0*area] = r / 255.0f;  // R
    dst[offset + 1*area] = g / 255.0f;  // G
    dst[offset + 2*area] = b / 255.0f;  // B
}

void cuda_preprocess_init(int max_w, int max_h)
{
    g_max_w = max_w;
    g_max_h = max_h;
    size_t bytes = (size_t)max_w * (size_t)max_h * 3;

    if (!g_src_device) {
        CHECK_CUDA(cudaMalloc(&g_src_device, bytes));
    }
}

void cuda_preprocess_destroy()
{
    if (g_src_device) {
        cudaFree(g_src_device);
        g_src_device = nullptr;
    }
    g_max_w = g_max_h = 0;
}

void cuda_preprocess(const uint8_t* src, int src_w, int src_h, int src_stride,
                     float* dst_chw, int dst_w, int dst_h,
                     cudaStream_t stream)
{
    if (!src || !dst_chw) return;
    if (!g_src_device) {
        std::cerr << "[preprocess] init not called\n";
        return;
    }
    if (src_w > g_max_w || src_h > g_max_h) {
        std::cerr << "[preprocess] src exceeds max buffer: "
                  << src_w << "x" << src_h << " > "
                  << g_max_w << "x" << g_max_h << "\n";
        return;
    }

    CHECK_CUDA(cudaMemcpy2DAsync(
        g_src_device,              // dst
        src_w * 3,                 // dst pitch (packed)
        src,                       // src
        src_stride,                // src pitch (cv::Mat.step)
        src_w * 3,                 // row bytes
        src_h,                     // rows
        cudaMemcpyHostToDevice,
        stream
    ));

    AffineMatrix d2s;
    float scale = fminf((float)dst_h / src_h, (float)dst_w / src_w);
    if (scale < 1e-6f) return;

    float tx = -scale * src_w * 0.5f + dst_w * 0.5f;
    float ty = -scale * src_h * 0.5f + dst_h * 0.5f;

    float inv_scale = 1.0f / scale;
    d2s.v[0] = inv_scale; d2s.v[1] = 0.0f;     d2s.v[2] = -tx * inv_scale;
    d2s.v[3] = 0.0f;     d2s.v[4] = inv_scale; d2s.v[5] = -ty * inv_scale;

    dim3 threads(32, 32);
    dim3 blocks((dst_w + threads.x - 1) / threads.x,
                (dst_h + threads.y - 1) / threads.y);

    warp_affine_bilinear_kernel<<<blocks, threads, 0, stream>>>(
        g_src_device, src_w * 3, src_w, src_h,
        dst_chw, dst_w, dst_h,
        114, d2s
    );
    CHECK_CUDA(cudaPeekAtLastError());
}
