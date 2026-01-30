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

__global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float conf_thresh, 
                              float* output, int max_objects, int* num_valid_objects) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bboxes) return;

    float max_conf = -1.0f;
    int max_id = -1;
    for (int c = 0; c < num_classes; c++) {
        float score = predict[(4 + c) * num_bboxes + i];
        if (score > max_conf) {
            max_conf = score;
            max_id = c;
        }
    }

    if (max_conf > conf_thresh) {
        int idx = atomicAdd(num_valid_objects, 1);
        if (idx >= max_objects) return;

        float cx = predict[0 * num_bboxes + i];
        float cy = predict[1 * num_bboxes + i];
        float w  = predict[2 * num_bboxes + i];
        float h  = predict[3 * num_bboxes + i];

        float left   = cx - w * 0.5f;
        float top    = cy - h * 0.5f;
        float right  = cx + w * 0.5f;
        float bottom = cy + h * 0.5f;

        int offset = idx * DATA_LENGTH; 
        output[offset + 0] = left;
        output[offset + 1] = top;
        output[offset + 2] = right;
        output[offset + 3] = bottom;
        output[offset + 4] = max_conf;
        output[offset + 5] = (float)max_id;
        output[offset + 6] = 1.0f; // 1.0 = Keep, 0.0 = Suppress
    }
}

__global__ void nms_kernel(float* bboxes, int* num_valid_objects, float nms_thresh, int max_objects) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int count = min(*num_valid_objects, max_objects);
    if (i >= count) return;

    int my_offset = i * DATA_LENGTH;
    float my_conf = bboxes[my_offset + 4];
    float my_class = bboxes[my_offset + 5];

    for (int j = 0; j < count; j++) {
        if (i == j) continue;

        int other_offset = j * DATA_LENGTH;
        float other_conf = bboxes[other_offset + 4];
        float other_class = bboxes[other_offset + 5];

        if (my_class != other_class) continue;

        if (other_conf > my_conf) {
            float iou = box_iou(bboxes[my_offset], bboxes[my_offset + 1], bboxes[my_offset + 2], bboxes[my_offset + 3],
                                bboxes[other_offset], bboxes[other_offset + 1], bboxes[other_offset + 2], bboxes[other_offset + 3]);
            if (iou > nms_thresh) {
                bboxes[my_offset + 6] = 0.0f;
                return;
            }
        }
        else if (other_conf == my_conf && j < i) {
             float iou = box_iou(bboxes[my_offset], bboxes[my_offset + 1], bboxes[my_offset + 2], bboxes[my_offset + 3],
                                bboxes[other_offset], bboxes[other_offset + 1], bboxes[other_offset + 2], bboxes[other_offset + 3]);
             if (iou > nms_thresh) {
                bboxes[my_offset + 6] = 0.0f;
                return;
             }
        }
    }
}

void cuda_decode(float* predict, int num_bboxes, int num_classes, float conf_thresh, 
                 float* output, int max_objects, int* num_valid_objects, cudaStream_t stream) 
{
    int block = 256;
    int grid = (num_bboxes + block - 1) / block;

    decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, conf_thresh, 
                                             output, max_objects, num_valid_objects);
}

void cuda_nms(float* output, int* num_valid_objects, float nms_thresh, int max_objects, cudaStream_t stream)
{
    int block = 256;
    int grid = (max_objects + block - 1) / block;
    
    nms_kernel<<<grid, block, 0, stream>>>(output, num_valid_objects, nms_thresh, max_objects);
}