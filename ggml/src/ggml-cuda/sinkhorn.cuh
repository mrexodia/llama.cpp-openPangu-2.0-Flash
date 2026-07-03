#include "common.cuh"

#define CUDA_SINKHORN_BLOCK_SIZE 128

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
