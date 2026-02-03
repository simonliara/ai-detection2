#include "cuda_utils.h"
#include "YOLOv11.h"

const int DATA_LENGTH = 7; 

__device__ float box_iou(float aleft, float atop, float aright, float abottom, 
                         float bleft, float btop, float bright, float bbottom) 
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void decode_kernel(const float* predict, int num_bboxes, int num_classes, 
                                        float conf_thresh, float* output, int max_objects, 
                                        int* num_valid_objects, float scale, float ox, float oy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bboxes) return;

    float best_conf = -1.0f;
    int best_id = -1;
    for (int c = 0; c < num_classes; c++) {
        float s = predict[(4 + c) * num_bboxes + i];
        if (s > best_conf) { best_conf = s; best_id = c; }
    }

    if (best_conf < conf_thresh) return;

    int idx = atomicAdd(num_valid_objects, 1);
    if (idx >= max_objects) return;

    float cx = predict[0 * num_bboxes + i];
    float cy = predict[1 * num_bboxes + i];
    float w  = predict[2 * num_bboxes + i];
    float h  = predict[3 * num_bboxes + i];

    float left   = (cx - 0.5f * w - ox) / scale;
    float top    = (cy - 0.5f * h - oy) / scale;
    float right  = (cx + 0.5f * w - ox) / scale;
    float bottom = (cy + 0.5f * h - oy) / scale;

    int off = idx * 7;
    output[off + 0] = left;
    output[off + 1] = top;
    output[off + 2] = right;
    output[off + 3] = bottom;
    output[off + 4] = best_conf;
    output[off + 5] = (float)best_id;
    output[off + 6] = 1.0f;
}

__global__ void nms_kernel(float* bboxes, int* num_valid_objects, float nms_thresh, int max_objects) 
{
    __shared__ float shared_boxes[256 * 4];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_count = min(*num_valid_objects, max_objects);
    if (i >= total_count) return;

    float ix1 = bboxes[i * 7 + 0];
    float iy1 = bboxes[i * 7 + 1];
    float ix2 = bboxes[i * 7 + 2];
    float iy2 = bboxes[i * 7 + 3];
    float iconf = bboxes[i * 7 + 4];
    float icls = bboxes[i * 7 + 5];

    for (int j = 0; j < total_count; j++) {
        if (i == j) continue;

        float jconf = bboxes[j * 7 + 4];
        float jcls = bboxes[j * 7 + 5];

        if (icls != jcls) continue;

        if (jconf > iconf || (jconf == iconf && j < i)) {
            float jx1 = bboxes[j * 7 + 0];
            float jy1 = bboxes[j * 7 + 1];
            float jx2 = bboxes[j * 7 + 2];
            float jy2 = bboxes[j * 7 + 3];

            float inter_x1 = max(ix1, jx1);
            float inter_y1 = max(iy1, jy1);
            float inter_x2 = min(ix2, jx2);
            float inter_y2 = min(iy2, jy2);
            
            float inter_w = max(0.0f, inter_x2 - inter_x1);
            float inter_h = max(0.0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;
            
            float area_i = (ix2 - ix1) * (iy2 - iy1);
            float area_j = (jx2 - jx1) * (jy2 - jy1);
            float iou = inter_area / (area_i + area_j - inter_area);

            if (iou > nms_thresh) {
                bboxes[i * 7 + 6] = 0.0f;
                return;
            }
        }
    }
}

void cuda_decode(const float* predict, int num_bboxes, int num_classes, 
                           float conf_thresh, float* output, int max_objects, 
                           int* num_valid_objects, float scale, float ox, float oy,
                           cudaStream_t stream) 
{
    int block = 256;
    int grid = (num_bboxes + block - 1) / block;

    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, conf_thresh, 
        output, max_objects, num_valid_objects, scale, ox, oy);
}

void cuda_nms(float* bboxes, int* num_valid_objects, float nms_thresh, 
                        int max_objects, cudaStream_t stream) 
{
    int block = 256;
    int grid = (max_objects + block - 1) / block;
    
    nms_kernel<<<grid, block, 0, stream>>>(
        bboxes, num_valid_objects, nms_thresh, max_objects);
}