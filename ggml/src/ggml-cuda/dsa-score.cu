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
