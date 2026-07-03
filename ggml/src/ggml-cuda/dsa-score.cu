#include "dsa-score.cuh"

// DSA lightning-indexer scores, one pass over the cached indexer keys:
//   out[s,t] = mask[s,t] + sum_h w[h,t] * relu(sum_d ik[d,s] * q[d,h,t])
// One warp per cached position s: the row load is coalesced across the lanes and
// the per-head dot products are warp-shuffle reductions. q and w for the block's
// token are staged in shared memory (d*h floats, <= 12 KB for the openPangu
// 128x24 indexer). Used on the decode path (nt small); prefill keeps the tiled
// matmul pipeline.
template <typename T_IK, typename T_MASK>
static __global__ void dsa_score_f32(
        const char * ik, const float * q, const float * w, const char * mask, float * dst,
        const int nd, const int64_t nkv, const int nh, const int64_t nb_ik_row, const int64_t nb_mask_row) {
    extern __shared__ float smem[];  // [nd*nh + nh]

    const int t    = blockIdx.y;
    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;

    const int     warps_per_block = blockDim.x/WARP_SIZE;
    const int64_t s = (int64_t) blockIdx.x*warps_per_block + warp;

    float * q_s = smem;
    float * w_s = smem + (size_t) nd*nh;

    for (int i = threadIdx.x; i < nd*nh; i += blockDim.x) {
        q_s[i] = q[(size_t) t*nd*nh + i];
    }
    for (int i = threadIdx.x; i < nh; i += blockDim.x) {
        w_s[i] = w[(size_t) t*nh + i];
    }
    __syncthreads();

    if (s >= nkv) {
        return;
    }

    const T_IK * ik_row = (const T_IK *) (ik + s*nb_ik_row);

    // coalesced row load: each lane keeps its d-strided slice in registers
    float ikl[8];  // nd <= 8*WARP_SIZE (256), asserted host-side
    const int per_lane = (nd + WARP_SIZE - 1)/WARP_SIZE;
    for (int i = 0; i < per_lane; ++i) {
        const int d = lane + i*WARP_SIZE;
        ikl[i] = d < nd ? (float) ik_row[d] : 0.0f;
    }

    float acc = 0.0f;
    for (int h = 0; h < nh; ++h) {
        const float * qh = q_s + h*nd;
        float part = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            const int d = lane + i*WARP_SIZE;
            if (d < nd) {
                part += ikl[i]*qh[d];
            }
        }
        part = warp_reduce_sum(part);
        acc += w_s[h]*fmaxf(part, 0.0f);
    }

    if (lane == 0) {
        const T_MASK * mask_row = (const T_MASK *) (mask + t*nb_mask_row);
        dst[(size_t) t*nkv + s] = acc + (float) mask_row[s];
    }
}

// Prefill variant: GEMM-like tiled kernel for large token counts. Each block
// computes a DSA_TILE_S x DSA_TILE_T output tile; the (transposed) ik tile is
// staged in shared memory once and reused across all heads, the per-head q tile
// is restaged each head iteration. Each thread accumulates a 4x2 register tile,
// so the shared-memory loads amortize over 8 FMAs. Never materializes the
// [n_kv, nh, nt] intermediate the unfused matmul pipeline needs (~5 GB per
// layer per 512-token ubatch at 100K context).
#define DSA_TILE_S       64
#define DSA_TILE_T       32
#define DSA_TILE_PS      (DSA_TILE_S + 4)  // padded rows: keeps 8-byte alignment, avoids bank conflicts
#define DSA_TILE_PT      (DSA_TILE_T + 2)
#define DSA_TILE_THREADS 256
#define DSA_TILE_ND_MAX  180               // smem = nd*(PS*2 + PT*4) bytes <= 48 KB

template <typename T_MASK>
static __global__ void dsa_score_tiled_f32(
        const char * ik, const float * q, const float * w, const char * mask, float * dst,
        const int nd, const int64_t nkv, const int nh, const int nt,
        const int64_t nb_ik_row, const int64_t nb_mask_row) {
    extern __shared__ char smem_raw[];
    half  * sik = (half  *)  smem_raw;                                          // [nd][DSA_TILE_PS], transposed
    float * sq  = (float *) (smem_raw + (size_t) nd*DSA_TILE_PS*sizeof(half));  // [nd][DSA_TILE_PT], transposed

    const int64_t s0 = (int64_t) blockIdx.x*DSA_TILE_S;
    const int     t0 = blockIdx.y*DSA_TILE_T;

    const int tid = threadIdx.x;
    const int tt  = tid % (DSA_TILE_T/2);  // covers t = t0 + tt*2 + {0,1}
    const int ts  = tid / (DSA_TILE_T/2);  // covers s = s0 + ts*4 + {0..3}

    // ik tile: coalesced global reads (d contiguous per row), transposed store
    for (int idx = tid; idx < nd*DSA_TILE_S; idx += DSA_TILE_THREADS) {
        const int d = idx % nd;
        const int s = idx / nd;
        sik[d*DSA_TILE_PS + s] = s0 + s < nkv ? ((const half *) (ik + (s0 + s)*nb_ik_row))[d] : __float2half(0.0f);
    }

    float acc[4][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int h = 0; h < nh; ++h) {
        __syncthreads();  // also covers the sik stores on the first iteration
        for (int idx = tid; idx < nd*DSA_TILE_T; idx += DSA_TILE_THREADS) {
            const int d = idx % nd;
            const int t = idx / nd;
            sq[d*DSA_TILE_PT + t] = t0 + t < nt ? q[((size_t) (t0 + t)*nh + h)*nd + d] : 0.0f;
        }
        __syncthreads();

        float dot[4][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        for (int d = 0; d < nd; ++d) {
            const half2 * a    = (const half2 *) (sik + d*DSA_TILE_PS + ts*4);
            const float2  a01  = __half22float2(a[0]);
            const float2  a23  = __half22float2(a[1]);
            const float2  b    = *(const float2 *) (sq + d*DSA_TILE_PT + tt*2);
            dot[0][0] += a01.x*b.x; dot[0][1] += a01.x*b.y;
            dot[1][0] += a01.y*b.x; dot[1][1] += a01.y*b.y;
            dot[2][0] += a23.x*b.x; dot[2][1] += a23.x*b.y;
            dot[3][0] += a23.y*b.x; dot[3][1] += a23.y*b.y;
        }

#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = t0 + tt*2 + j;
            const float wv = t < nt ? w[(size_t) t*nh + h] : 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i][j] += wv*fmaxf(dot[i][j], 0.0f);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int t = t0 + tt*2 + j;
        if (t >= nt) {
            continue;
        }
        const T_MASK * mask_row = (const T_MASK *) (mask + (size_t) t*nb_mask_row);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t s = s0 + ts*4 + i;
            if (s < nkv) {
                dst[(size_t) t*nkv + s] = acc[i][j] + (float) mask_row[s];
            }
        }
    }
}

void ggml_cuda_op_dsa_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ik   = dst->src[0];
    const ggml_tensor * q    = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32);

    const int     nd  = (int) ik->ne[0];
    const int64_t nkv = ik->ne[1];
    const int     nh  = (int) q->ne[1];
    const int     nt  = (int) q->ne[2];

    GGML_ASSERT(nd <= 8*WARP_SIZE);

    // large token counts (prefill): tiled kernel, one output tile per block
    if (nt >= 16 && ik->type == GGML_TYPE_F16 && nd <= DSA_TILE_ND_MAX) {
        const dim3 grid((nkv + DSA_TILE_S - 1)/DSA_TILE_S, (nt + DSA_TILE_T - 1)/DSA_TILE_T, 1);
        const size_t smem = (size_t) nd*(DSA_TILE_PS*sizeof(half) + DSA_TILE_PT*sizeof(float));

        if (mask->type == GGML_TYPE_F16) {
            dsa_score_tiled_f32<half><<<grid, DSA_TILE_THREADS, smem, ctx.stream()>>>(
                    (const char *) ik->data, (const float *) q->data, (const float *) w->data,
                    (const char *) mask->data, (float *) dst->data,
                    nd, nkv, nh, nt, ik->nb[1], mask->nb[1]);
        } else {
            dsa_score_tiled_f32<float><<<grid, DSA_TILE_THREADS, smem, ctx.stream()>>>(
                    (const char *) ik->data, (const float *) q->data, (const float *) w->data,
                    (const char *) mask->data, (float *) dst->data,
                    nd, nkv, nh, nt, ik->nb[1], mask->nb[1]);
        }
        return;
    }

    const int warps_per_block = CUDA_DSA_SCORE_BLOCK_SIZE/WARP_SIZE;
    const dim3 grid((nkv + warps_per_block - 1)/warps_per_block, nt, 1);
    const size_t smem = ((size_t) nd*nh + nh)*sizeof(float);

    const auto launch = [&](auto ik_dummy, auto mask_dummy) {
        using T_IK   = decltype(ik_dummy);
        using T_MASK = decltype(mask_dummy);
        dsa_score_f32<T_IK, T_MASK><<<grid, CUDA_DSA_SCORE_BLOCK_SIZE, smem, ctx.stream()>>>(
                (const char *) ik->data, (const float *) q->data, (const float *) w->data,
                (const char *) mask->data, (float *) dst->data,
                nd, nkv, nh, ik->nb[1], mask->nb[1]);
    };

    if (ik->type == GGML_TYPE_F16) {
        if (mask->type == GGML_TYPE_F16) {
            launch(half{}, half{});
        } else {
            launch(half{}, float{});
        }
    } else {
        if (mask->type == GGML_TYPE_F16) {
            launch(float{}, half{});
        } else {
            launch(float{}, float{});
        }
    }
}
