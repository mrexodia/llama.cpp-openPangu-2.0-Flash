#include "sinkhorn.cuh"

// Shared by GGML_OP_SINKHORN and GGML_OP_HC_MIX: softmax along dim0, add eps,
// then alternating dim1/dim0 normalization of the [ne0, ne1] matrix m.
// The matrices are tiny (hyper-connection stream mixing, ne0*ne1 <= 64), so one
// thread handles one matrix entirely in local memory. This replaces the
// ~8 ops x 2 x n_iter graph nodes the unfused form needs per matrix.
static __device__ void sinkhorn_matrix(float * m, const int ne0, const int ne1, const int n_iter, const float eps) {
    const int nm = ne0*ne1;

    // softmax along dim0, per i1
    for (int i1 = 0; i1 < ne1; ++i1) {
        float * c = m + i1*ne0;
        float vmax = c[0];
        for (int i0 = 1; i0 < ne0; ++i0) {
            vmax = fmaxf(vmax, c[i0]);
        }
        float sum = 0.0f;
        for (int i0 = 0; i0 < ne0; ++i0) {
            c[i0] = expf(c[i0] - vmax);
            sum += c[i0];
        }
        // reciprocal multiply to match ggml_soft_max rounding
        const float inv = 1.0f/sum;
        for (int i0 = 0; i0 < ne0; ++i0) {
            c[i0] *= inv;
        }
    }

    // add eps to every element
    for (int i = 0; i < nm; ++i) {
        m[i] += eps;
    }

    for (int it = 0; it < n_iter; ++it) {
        // normalize along dim0, per i1 (skipped on the first pass)
        if (it > 0) {
            for (int i1 = 0; i1 < ne1; ++i1) {
                float sum = eps;
                for (int i0 = 0; i0 < ne0; ++i0) {
                    sum += m[i0 + i1*ne0];
                }
                for (int i0 = 0; i0 < ne0; ++i0) {
                    m[i0 + i1*ne0] /= sum;
                }
            }
        }
        // normalize along dim1, per i0
        for (int i0 = 0; i0 < ne0; ++i0) {
            float sum = eps;
            for (int i1 = 0; i1 < ne1; ++i1) {
                sum += m[i0 + i1*ne0];
            }
            for (int i1 = 0; i1 < ne1; ++i1) {
                m[i0 + i1*ne0] /= sum;
            }
        }
    }
}

static __global__ void sinkhorn_f32(const float * x, float * dst,
        const int ne0, const int ne1, const int64_t nb, const int n_iter, const float eps) {
    const int64_t b = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nb) {
        return;
    }

    const int nm = ne0*ne1;

    float m[64];

    const float * sp = x   + b*nm;
          float * dp = dst + b*nm;

    for (int i = 0; i < nm; ++i) {
        m[i] = sp[i];
    }

    sinkhorn_matrix(m, ne0, ne1, n_iter, eps);

    for (int i = 0; i < nm; ++i) {
        dp[i] = m[i];
    }
}

static __global__ void hc_mix_f32(const float * x, const float * scale, const float * base, float * dst,
        const int hc, const int64_t nt, const int n_iter, const float eps) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nt) {
        return;
    }

    const int n0 = 2*hc + hc*hc;

    float m[64];

    const float * sp = x   + t*n0;
          float * dp = dst + t*n0;

    for (int i = 0; i < hc; ++i) {
        dp[i] = 1.0f/(1.0f + expf(-(sp[i]*scale[0] + base[i]))) + eps;   // pre
    }
    for (int i = hc; i < 2*hc; ++i) {
        dp[i] = 2.0f/(1.0f + expf(-(sp[i]*scale[1] + base[i])));         // post
    }
    for (int i = 2*hc; i < n0; ++i) {
        m[i - 2*hc] = sp[i]*scale[2] + base[i];                          // comb
    }

    sinkhorn_matrix(m, hc, hc, n_iter, eps);

    for (int i = 0; i < hc*hc; ++i) {
        dp[2*hc + i] = m[i];
    }
}

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[0]*src0->ne[1] <= 64);

    const int   n_iter = ggml_get_op_params_i32(dst, 0);
    const float eps    = ggml_get_op_params_f32(dst, 1);

    const int64_t nb = src0->ne[2]*src0->ne[3];

    const int64_t num_blocks = (nb + CUDA_SINKHORN_BLOCK_SIZE - 1)/CUDA_SINKHORN_BLOCK_SIZE;

    sinkhorn_f32<<<num_blocks, CUDA_SINKHORN_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const float *) src0->data, (float *) dst->data,
            (int) src0->ne[0], (int) src0->ne[1], nb, n_iter, eps);
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0  = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int   hc     = ggml_get_op_params_i32(dst, 0);
    const int   n_iter = ggml_get_op_params_i32(dst, 1);
    const float eps    = ggml_get_op_params_f32(dst, 2);

    GGML_ASSERT(src0->ne[0] == 2*hc + hc*hc);
    GGML_ASSERT(hc*hc <= 64);

    const int64_t nt = src0->ne[1]*src0->ne[2]*src0->ne[3];

    const int64_t num_blocks = (nt + CUDA_SINKHORN_BLOCK_SIZE - 1)/CUDA_SINKHORN_BLOCK_SIZE;

    hc_mix_f32<<<num_blocks, CUDA_SINKHORN_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const float *) src0->data, (const float *) scale->data, (const float *) base->data,
            (float *) dst->data, hc, nt, n_iter, eps);
}
