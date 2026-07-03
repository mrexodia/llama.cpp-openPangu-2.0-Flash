#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

// map float bits to a monotonically ordered uint (larger float <=> larger uint)
static __device__ __forceinline__ uint32_t top_k_float_to_ordered(const float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// radix-select top-k, one block per row: 4 rounds of 256-bin byte histograms
// (MSB first) narrow down the exact bit pattern of the k-th largest value, then
// a single compaction pass collects the winning indices. O(n) passes instead of
// a full O(n log n) sort -- at [102400, 512] this is ~5 reads of the row data
// vs a full segmented radix sort. Output order is unspecified, which matches
// the GGML_OP_TOP_K contract.
#define CUDA_TOP_K_RADIX_BLOCK_SIZE 512

static __global__ void top_k_radix_select_f32(const float * src, int * dst, const int ncols, const int k) {
    const float * x   = src + (size_t) blockIdx.x*ncols;
    int         * out = dst + (size_t) blockIdx.x*k;

    __shared__ int      hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_remaining;  // elements of the current prefix group still needed
    __shared__ int      s_pos_gt, s_pos_eq;

    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix    = 0;
        s_remaining = k;
    }

    for (int round = 0; round < 4; ++round) {
        const int shift = (3 - round)*8;

        for (int i = tid; i < 256; i += blockDim.x) {
            hist[i] = 0;
        }
        __syncthreads();

        const uint32_t prefix      = s_prefix;
        const uint32_t prefix_mask = round == 0 ? 0 : 0xFFFFFFFFu << (shift + 8);
        for (int i = tid; i < ncols; i += blockDim.x) {
            const uint32_t u = top_k_float_to_ordered(x[i]);
            if ((u & prefix_mask) == prefix) {
                const int bin = (u >> shift) & 255;
                // warp-aggregate equal bins before hitting shared memory: masked-out
                // rows produce millions of identical -inf entries otherwise
                const unsigned int peers = __match_any_sync(__activemask(), bin);
                if ((__ffs(peers) - 1) == (int) (threadIdx.x % WARP_SIZE)) {
                    atomicAdd(&hist[bin], __popc(peers));
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            int rem = s_remaining;
            int b   = 255;
            while (b > 0 && hist[b] < rem) {
                rem -= hist[b];
                --b;
            }
            s_prefix    = prefix | ((uint32_t) b << shift);
            s_remaining = rem;
        }
        __syncthreads();
    }

    const uint32_t T    = s_prefix;          // ordered bits of the k-th largest value
    const int      n_gt = k - s_remaining;   // elements strictly greater than T

    if (tid == 0) {
        s_pos_gt = 0;
        s_pos_eq = 0;
    }
    __syncthreads();

    for (int i = tid; i < ncols; i += blockDim.x) {
        const uint32_t u = top_k_float_to_ordered(x[i]);
        if (u > T) {
            out[atomicAdd(&s_pos_gt, 1)] = i;
        } else if (u == T) {
            const int slot = atomicAdd(&s_pos_eq, 1);
            if (slot < k - n_gt) {
                out[n_gt + slot] = i;
            }
        }
    }
}

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    // large batched rows: radix select, O(n) instead of a full sort (order is
    // unspecified). single rows stay on the sort paths -- one block per row
    // leaves the GPU idle at nrows == 1
    if (ncols >= 4096 && nrows > 1) {
        top_k_radix_select_f32<<<nrows, CUDA_TOP_K_RADIX_BLOCK_SIZE, 0, stream>>>(
                src0_d, dst_d, ncols, k);
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
