#include "common.cuh"

#define CUDA_DSA_SCORE_BLOCK_SIZE 128

void ggml_cuda_op_dsa_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
