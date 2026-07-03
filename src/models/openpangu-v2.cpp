#include "models.h"

#include "ggml-cpp.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

// openPangu-2.0-Flash (openpangu-v2)
//
// Composition of existing infra:
//   - 4-stream mHC residual (manifold hyper-connections), same family as DeepSeek-V4.
//     The learned RMSNorm gamma is folded into the phi weight at conversion time.
//   - MLA attention (absorbed), same as DeepSeek-V2, scale 1/sqrt(qk_head_dim).
//   - MoME: width-k causal depthwise conv + identity residual on q_a, kv-compressed, and attn-out.
//   - 128 learned param-sink KV entries per layer (always visible, not indexer-scored).
//   - Sigmoid-gated MoE with expert bias + shared expert, leading dense layers.
//   - DSA lightning indexer on the non-SWA trunk layers: per-token top-k selection over
//     the cached tokens, applied as a mask on the full causal attention. The indexer key
//     rides in the tail of the cached MLA K row (no separate indexer cache).
//
// The MoME conv state is carried across ubatches via the recurrent memory; SWA layers use
// the iSWA split cache.

static ggml_tensor * opv2_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, i0 * ggml_element_size(t));
}

static ggml_tensor * opv2_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], i0 * ggml_element_size(t));
}

static ggml_tensor * opv2_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

void llama_model_openpangu_v2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.n_layer_nextn, false);

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl, false);

    // DSA lightning indexer (top-k token selection on the non-SWA trunk layers)
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k, false);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    // mHC (reuse the DeepSeek-V4 hyper-connection fields)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    // required: the param-sink and MoME conv tensors are sized from these
    ml.get_key(LLM_KV_ATTENTION_SINK_COUNT,       hparams.openpangu_n_sink);
    ml.get_key(LLM_KV_ATTENTION_CONV_KERNEL_SIZE, hparams.openpangu_conv_k);

    // the param sinks live in reserved rows at the head of the K cache; pad the
    // prefix to the flash-attention stride so the total KV length stays aligned
    hparams.n_k_sink_prefix = GGML_PAD(hparams.openpangu_n_sink, 256);

    // per-layer sliding window: SWA layers use a local window, DSA layers are global.
    // Both read from the same compressed MLA KV cache; the iSWA split keeps the
    // windowed layers cheap in memory at long context.
    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_window;
    swa_window.fill(0);
    ml.get_arr(LLM_KV_ATTENTION_LAYER_SWA_WINDOW, swa_window, false);
    uint32_t swa = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = swa_window[il] != 0;
        if (swa_window[il] != 0) {
            swa = swa_window[il];
        }
    }
    if (swa == 0) {
        // the graph and memory creation assume the hybrid-iswa layout
        throw std::runtime_error("openpangu-v2 requires attention.layer_swa_window with at least one non-zero window");
    }
    hparams.n_swa    = swa;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // MLA stores a single compressed KV head: size the KV cache to the
    // compressed dims (kv_lora + rope for K, kv_lora for V). The real
    // per-head attention dims come from n_embd_head_{k,v}_mla().
    // DSA layers append their indexer key to the same cache row; layers
    // without an indexer (SWA, MTP) zero-pad the tail.
    hparams.n_embd_head_k_full = hparams.n_lora_kv + hparams.n_rot() + hparams.indexer_head_size;
    hparams.n_embd_head_v_full = hparams.n_lora_kv;
    // SWA layers share the same compressed MLA KV geometry (used to size the iSWA SWA sub-cache)
    hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
    hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
    hparams.n_rot_swa         = hparams.n_rot();

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_openpangu_v2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank   = hparams.n_lora_q;
    const int64_t kv_lora_rank  = hparams.n_lora_kv;
    const int64_t n_ff_exp      = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t hc            = hparams.dsv4_hc_mult;
    const int64_t conv_k        = hparams.openpangu_conv_k;
    const int64_t n_sink        = hparams.openpangu_n_sink;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // model-level mHC stream merge (pre_only); absent in split MTP-only GGUFs
    const int mhc_flags = n_layer > 0 ? 0 : TENSOR_NOT_REQUIRED;
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc * n_embd, hc}, mhc_flags);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc}, mhc_flags);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, mhc_flags);

    const int64_t hc_mix_dim = (2 + hc) * hc;

    for (int i = 0; i < (int) hparams.n_layer_all; ++i) {
        auto & layer = layers[i];
        const bool is_mtp = i >= n_layer;   // MTP / NextN heads have no mHC or indexer

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, 0);

        if (!is_mtp) {
            layer.block_post_norm = create_tensor(tn(LLM_TENSOR_BLOCK_POST_NORM, "weight", i), {hc * n_embd}, TENSOR_NOT_REQUIRED);

            // mHC per-layer
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc * n_embd, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc * n_embd, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);
        } else {
            // MTP / NextN head projections (shared_head/embed fall back to model.tok_embd/output)
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, 0);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, 0);
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, 0);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        }

        // MLA
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
        layer.wq_a      = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,     "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b      = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,     "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,"weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B,     "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B,     "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,     "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        // MoME convs (stored [k, dim])
        layer.attn_q_a_conv  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_CONV,  "weight", i), {conv_k, q_lora_rank}, 0);
        layer.attn_kv_a_conv = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_CONV, "weight", i), {conv_k, kv_lora_rank}, 0);
        layer.attn_o_conv    = create_tensor(tn(LLM_TENSOR_ATTN_O_CONV,    "weight", i), {conv_k, n_head * n_embd_head_v_mla}, 0);

        // learned param sinks
        layer.attn_sink_kv   = create_tensor(tn(LLM_TENSOR_ATTN_SINK_KV,   "weight", i), {kv_lora_rank, n_sink}, 0);
        layer.attn_sink_k_pe = create_tensor(tn(LLM_TENSOR_ATTN_SINK_K_PE, "weight", i), {n_embd_head_qk_rope, n_sink}, 0);

        // DSA lightning indexer (present only on the non-SWA trunk layers)
        const int64_t idx_h = hparams.indexer_n_head;
        const int64_t idx_d = hparams.indexer_head_size;
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, idx_h * idx_d}, TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, idx_d}, TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {idx_d}, TENSOR_NOT_REQUIRED);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, idx_h}, TENSOR_NOT_REQUIRED);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_sink_mask_blk(ggml_tensor * kq_mask) const {
    const int64_t n_sink = hparams.openpangu_n_sink;
    const int64_t n_pfx  = hparams.n_k_sink_prefix;

    ggml_tensor * m_vis = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sink,
            kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
    m_vis = ggml_fill(ctx0, m_vis, 0.0f);                              // real sinks: always visible
    ggml_tensor * blk = m_vis;
    if (n_pfx > n_sink) {
        ggml_tensor * m_pad = ggml_new_tensor_4d(ctx0, kq_mask->type, n_pfx - n_sink,
                kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
        m_pad = ggml_fill(ctx0, m_pad, -INFINITY);                     // padding rows: masked out
        blk = ggml_concat(ctx0, m_vis, m_pad, 0);
    }
    return blk;
}

void llama_model_openpangu_v2::graph_base::build_sink_write(const llama_layer & layer, ggml_tensor * k_row, int il) const {
    const int64_t n_sink = hparams.openpangu_n_sink;
    const int64_t n_pfx  = hparams.n_k_sink_prefix;
    const int64_t idx_d  = hparams.indexer_head_size;

    // sink rows [row_width, n_pfx]: RMS-norm'd compressed KV + k_pe (not RoPE'd),
    // zero indexer tail and zero (masked) padding rows. The values are static;
    // rewriting the small prefix every graph avoids a load-time hook.
    ggml_tensor * sink_in = ggml_cast(ctx0, layer.attn_sink_kv, GGML_TYPE_F32);
    ggml_tensor * sink_kv = build_norm(sink_in, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * sink_pe = ggml_cast(ctx0, layer.attn_sink_k_pe, sink_kv->type);
    ggml_tensor * sink_k  = ggml_concat(ctx0, sink_kv, sink_pe, 0);
    if (idx_d > 0 || n_pfx > n_sink) {
        sink_k = ggml_pad(ctx0, sink_k, idx_d, n_pfx - n_sink, 0, 0);
    }
    sink_k = ggml_cast(ctx0, sink_k, k_row->type);

    ggml_tensor * dst = ggml_view_2d(ctx0, k_row, k_row->ne[0], n_pfx, k_row->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, sink_k, dst));
}

// The mHC stream reductions have two shapes: a batched-matmul form (few kernel
// launches -- wins at decode, where launch overhead dominates) and a per-stream
// loop form (no stream-transpose copies or tiny batched GEMMs -- wins at prefill).
// The transposed stream state xt selects the form: non-null => batched.
static const int64_t OPV2_HC_BATCHED_NT_MAX = 8;

ggml_tensor * llama_model_openpangu_v2::graph_base::build_hc_weighted_sum(ggml_tensor * x, ggml_tensor * xt, ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (xt) {
        // out[e,t] = sum_h xt[h,e,t] * w[h,t]: one batched matvec over the streams
        if (!ggml_is_contiguous(weights)) {
            weights = ggml_cont(ctx0, weights);                    // hc_mix output views are strided
        }
        ggml_tensor * wr  = ggml_reshape_3d(ctx0, weights, hc, 1, nt);
        ggml_tensor * out = ggml_mul_mat(ctx0, xt, wr);            // [n_embd, 1, nt]
        return ggml_reshape_2d(ctx0, out, n_embd, nt);
    }

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih * x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih * weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }
    return acc;
}

bool llama_model_openpangu_v2::sinkhorn_fused() const {
    if (sinkhorn_fused_probe < 0) {
        sinkhorn_fused_probe = 1;

        for (const llama_device & ldev : devices) {
            if (ldev.is_meta) {
                continue;
            }
            ggml_backend_dev_t dev = ldev.dev;

            ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead()*16,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx { ggml_init(params) };
            GGML_ASSERT(ctx);

            ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
            ggml_backend_buffer_ptr    buf  { ggml_backend_buft_alloc_buffer(buft, 0) };

            ggml_tensor * a     = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 4, 1);
            ggml_tensor * mixes = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 24, 1);
            ggml_tensor * sc    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
            ggml_tensor * bs    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 24);
            ggml_tensor * ik    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 128, 16);
            ggml_tensor * dq    = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 24, 1);
            ggml_tensor * dw    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 24, 1);
            ggml_tensor * dm    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 16, 1);

            ggml_tensor * op_sinkhorn  = ggml_sinkhorn(ctx.get(), a, 20, 1e-6f);
            ggml_tensor * op_hc_mix    = ggml_hc_mix(ctx.get(), mixes, sc, bs, 4, 20, 1e-6f);
            ggml_tensor * op_dsa_score = ggml_dsa_score(ctx.get(), ik, dq, dw, dm);

            a->buffer     = buf.get();
            mixes->buffer = buf.get();
            sc->buffer    = buf.get();
            bs->buffer    = buf.get();
            ik->buffer    = buf.get();
            dq->buffer    = buf.get();
            dw->buffer    = buf.get();
            dm->buffer    = buf.get();

            if (!ggml_backend_dev_supports_op(dev, op_sinkhorn) ||
                !ggml_backend_dev_supports_op(dev, op_hc_mix)   ||
                !ggml_backend_dev_supports_op(dev, op_dsa_score)) {
                LLAMA_LOG_WARN("%s: device %s does not support the fused hyper-connection ops - using the unfused op sequence\n",
                        __func__, ggml_backend_dev_name(dev));
                sinkhorn_fused_probe = 0;
                break;
            }
        }
    }
    return sinkhorn_fused_probe != 0;
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_hc_sinkhorn(ggml_tensor * comb, int il) const {
    GGML_UNUSED(il);

    if (use_fused_sinkhorn) {
        // fused softmax + eps + alternating row/column normalization: one op instead
        // of the ~8 nodes x 2 x n_iter the unfused form needs per [hc, hc] matrix
        return ggml_sinkhorn(ctx0, comb, hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);
    }

    // unfused fallback for backends without GGML_OP_SINKHORN (numerically equivalent)
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);
    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };
    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }
    return comb;
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_hc_pre(ggml_tensor * x, ggml_tensor * xt,
        ggml_tensor * hc_fn, ggml_tensor * hc_scale,
        ggml_tensor * hc_base, ggml_tensor ** post, ggml_tensor ** comb, int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc * n_embd;
    const int64_t nt         = x->ne[2];

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);

    if (use_fused_sinkhorn) {
        // fused affine + sigmoid gates + sinkhorn: one op instead of ~12 per call
        ggml_tensor * mixed = ggml_hc_mix(ctx0, mixes, hc_scale, hc_base,
                (int) hc, (int) hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);

        const size_t es = ggml_element_size(mixed);
        ggml_tensor * pre = ggml_view_2d(ctx0, mixed, hc, nt, mixed->nb[1], 0);
        *post = ggml_view_2d(ctx0, mixed, hc, nt, mixed->nb[1], hc*es);
        *comb = ggml_view_3d(ctx0, mixed, hc, hc, nt, hc*es, mixed->nb[1], 2*hc*es);

        return build_hc_weighted_sum(x, xt, pre);
    }

    ggml_tensor * scale_pre  = opv2_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = opv2_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = opv2_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = opv2_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = opv2_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = opv2_view_1d(ctx0, hc_base, hc * hc, 2 * hc);

    ggml_tensor * pre = opv2_view_2d(ctx0, mixes, hc, nt, 0);
    pre = opv2_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);

    *post = opv2_view_2d(ctx0, mixes, hc, nt, hc);
    *post = opv2_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);

    *comb = opv2_view_2d(ctx0, mixes, hc * hc, nt, 2 * hc);
    *comb = opv2_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb, il);

    return build_hc_weighted_sum(x, xt, pre);
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_hc_post(ggml_tensor * x, ggml_tensor * residual,
        ggml_tensor * residual_t, ggml_tensor * post, ggml_tensor * comb, int il) const {
    GGML_UNUSED(il);
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    if (residual_t) {
        // merged[e,dst,t] = sum_src residual_t[src,e,t] * comb[dst,src,t]: batched matmul.
        // comb is laid out [dst, src, t]; mul_mat contracts dim0, so transpose the
        // (tiny) mixing matrix to [src, dst, t] first
        ggml_tensor * comb_t = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * merged = ggml_mul_mat(ctx0, residual_t, comb_t); // [n_embd, hc, nt]

        // xpost[e,dst,t] = x[e,t] * post[dst,t]: rank-1 outer product per token
        if (!ggml_is_contiguous(post)) {
            post = ggml_cont(ctx0, post);                              // hc_mix output views are strided
        }
        ggml_tensor * xr    = ggml_reshape_3d(ctx0, x, 1, n_embd, nt);
        ggml_tensor * pr    = ggml_reshape_3d(ctx0, post, 1, hc, nt);
        ggml_tensor * xpost = ggml_mul_mat(ctx0, xr, pr);              // [n_embd, hc, nt]

        return ggml_add(ctx0, merged, xpost);
    }

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst * post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);
        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src * residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst * comb->nb[0] + src * comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }
    return out;
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_hc_head(ggml_tensor * x, ggml_tensor * hc_fn,
        ggml_tensor * hc_scale, ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);

    // pre_only merge: plain sigmoid, no eps (the reference only adds hc_eps on
    // the full pre/post/comb path, not in the pre_only head)
    ggml_tensor * pre = opv2_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);

    ggml_tensor * xt = nt <= OPV2_HC_BATCHED_NT_MAX
        ? ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3))                // [hc, n_embd, nt]
        : nullptr;
    return build_hc_weighted_sum(x, xt, pre);
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_mome_conv(ggml_tensor * x, ggml_tensor * conv_w,
        ggml_tensor * conv_rs, ggml_tensor * conv_state, int64_t rs_head,
        int64_t state_off, int64_t n_seqs, int64_t n_seq_tokens) const {
    // x: [dim, n_tokens]; conv_w: [k, dim] (bf16 in gguf, cast to f32 for ssm_conv).
    // Causal depthwise conv with an identity residual. The previous (k-1) input
    // columns are carried across ubatches via the recurrent state.
    const int64_t dim    = x->ne[0];
    const int64_t nt     = x->ne[1];
    const int64_t k      = conv_w->ne[0];
    const int64_t d_conv = k - 1;                         // state width per channel
    const int64_t esz    = ggml_element_size(conv_state);

    ggml_tensor * w = conv_w->type == GGML_TYPE_F32 ? conv_w : ggml_cast(ctx0, conv_w, GGML_TYPE_F32);

    // previous-token state for this conv: strided view of the gathered recurrent state
    ggml_tensor * state = ggml_view_3d(ctx0, conv_rs, d_conv, dim, n_seqs,
            d_conv * esz, conv_rs->nb[1], state_off * esz);                       // [k-1, dim, n_seqs]

    // input as [n_seq_tokens, dim, n_seqs]; concat handles the non-contiguous permute
    ggml_tensor * xt = ggml_reshape_3d(ctx0, x, dim, n_seq_tokens, n_seqs);
    xt = ggml_permute(ctx0, xt, 1, 0, 2, 3);                                      // [n_seq_tokens, dim, n_seqs]

    ggml_tensor * sx = ggml_concat(ctx0, state, xt, 0);                           // [k-1+n_seq_tokens, dim, n_seqs]

    // store the last (k-1) input columns as the next state
    ggml_tensor * new_state = ggml_view_3d(ctx0, sx, d_conv, dim, n_seqs, sx->nb[1], sx->nb[2],
            n_seq_tokens * sx->nb[0]);
    ggml_tensor * dst = ggml_view_2d(ctx0, conv_state, d_conv * dim, n_seqs, conv_state->nb[1],
            rs_head * conv_state->nb[1] + state_off * esz);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, ggml_cont(ctx0, new_state), d_conv * dim, n_seqs), dst));

    ggml_tensor * conv = ggml_ssm_conv(ctx0, sx, w);                              // [dim, n_seq_tokens, n_seqs]
    conv = ggml_reshape_2d(ctx0, conv, dim, nt);

    return ggml_add(ctx0, x, conv);                                              // identity residual
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_attention(const llama_model & model, llm_graph_input_mem_hybrid_iswa * inp,
        ggml_tensor * cur, ggml_tensor * inp_pos, float kq_scale, int il) const {
    const auto & layer = model.layers[il];
    auto * inp_attn = inp->get_attn();
    const bool is_swa = hparams.is_swa(il);

    const int64_t n_embd_head_k      = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t nt                  = cur->ne[1];

    // recurrent MoME conv state (previous k-1 input columns per conv, per sequence)
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t d_conv       = hparams.openpangu_conv_k - 1;
    const int64_t off_qa       = 0;
    const int64_t off_kv       = d_conv * hparams.n_lora_q;
    const int64_t off_o        = d_conv * (hparams.n_lora_q + hparams.n_lora_kv);

    const auto * rmctx = static_cast<const llama_memory_hybrid_iswa_context *>(mctx)->get_recr();
    ggml_tensor * conv_state = rmctx->get_r_l(il);
    const int64_t rs_head    = rmctx->get_head();
    ggml_tensor * conv_rs    = build_rs(inp->get_recr(), conv_state, hparams.n_embd_r(), n_seqs);

    // q down -> conv -> norm -> up
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    q = build_mome_conv(q, layer.attn_q_a_conv, conv_rs, conv_state, rs_head, off_qa, n_seqs, n_seq_tokens);
    ggml_tensor * q_lora = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_mul_mat(ctx0, layer.wq_b, q_lora);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    // kv down: split compressed (kv_lora) and rope (k_pe); conv applies to compressed only
    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, nt,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, nt,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    kv_cmpr = ggml_cont(ctx0, kv_cmpr);
    kv_cmpr = build_mome_conv(kv_cmpr, layer.attn_kv_a_conv, conv_rs, conv_state, rs_head, off_kv, n_seqs, n_seq_tokens);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_qk_rope, GGML_ROPE_TYPE_NEOX,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_embd_head_qk_rope, GGML_ROPE_TYPE_NEOX,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // absorb q_nope into the compressed space via wk_b
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);                     // [nope, nt, n_head]
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);   // [kv_lora, n_head, nt]

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);    // [kv_lora+rope, n_head, nt]

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, nt);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);            // [kv_lora+rope, 1, nt]

    // DSA lightning indexer: the per-token indexer key rides in the tail of the
    // cached K row; layers without an indexer (SWA) zero-pad the tail instead
    const int64_t idx_d  = hparams.indexer_head_size;
    const int64_t idx_h  = hparams.indexer_n_head;
    const bool    is_dsa = idx_d > 0 && hparams.indexer_top_k > 0 && !is_swa &&
                           layer.indexer_attn_q_b && layer.indexer_attn_k && layer.indexer_proj;

    ggml_tensor * indexer_q = nullptr;
    if (is_dsa) {
        // query heads from the post-conv, post-norm q_lora (the wq_b input);
        // rope on the leading n_rot dims of each head, as in the main attention
        indexer_q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, q_lora);   // [idx_h*idx_d, nt]
        ggml_tensor * iq_pe = ggml_view_3d(ctx0, indexer_q, n_embd_head_qk_rope, idx_h, nt,
                ggml_row_size(indexer_q->type, idx_d), ggml_row_size(indexer_q->type, idx_d) * idx_h, 0);
        ggml_tensor * iq_nope = ggml_view_3d(ctx0, indexer_q, idx_d - n_embd_head_qk_rope, idx_h, nt,
                ggml_row_size(indexer_q->type, idx_d), ggml_row_size(indexer_q->type, idx_d) * idx_h,
                ggml_row_size(indexer_q->type, n_embd_head_qk_rope));
        iq_pe = ggml_rope_ext(ctx0, iq_pe, inp_pos, nullptr, n_embd_head_qk_rope, GGML_ROPE_TYPE_NEOX,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        indexer_q = ggml_concat(ctx0, iq_pe, iq_nope, 0);                 // [idx_d, idx_h, nt]

        // single shared key from the attention input: RMS norm, same rope split
        ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur); // [idx_d, nt]
        ik = build_norm(ik, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * ik_pe = ggml_view_3d(ctx0, ik, n_embd_head_qk_rope, 1, nt,
                ggml_row_size(ik->type, idx_d), ggml_row_size(ik->type, idx_d), 0);
        ggml_tensor * ik_nope = ggml_view_3d(ctx0, ik, idx_d - n_embd_head_qk_rope, 1, nt,
                ggml_row_size(ik->type, idx_d), ggml_row_size(ik->type, idx_d),
                ggml_row_size(ik->type, n_embd_head_qk_rope));
        ik_pe = ggml_rope_ext(ctx0, ik_pe, inp_pos, nullptr, n_embd_head_qk_rope, GGML_ROPE_TYPE_NEOX,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        ik = ggml_concat(ctx0, ik_pe, ik_nope, 0);                        // [idx_d, 1, nt]

        Kcur = ggml_concat(ctx0, Kcur, ik, 0);                            // [kv_lora+rope+idx_d, 1, nt]
    } else if (idx_d > 0) {
        Kcur = ggml_pad(ctx0, Kcur, idx_d, 0, 0, 0);
    }

    // store compressed K to the right iSWA sub-cache (full for DSA/global layers,
    // windowed for SWA layers). MLA: V is a view of K, decompressed by wv_b.
    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Kcur);
    const auto * attn_mctx = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
    ggml_tensor * k_idxs = is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
    ggml_build_forward_expand(gf, attn_mctx->cpy_k(ctx0, Kcur, k_idxs, il));

    const int64_t n_pfx = hparams.n_k_sink_prefix;

    ggml_tensor * k_row = attn_mctx->get_k(ctx0, il);                    // [kv_lora+rope(+idx_d), 1, n_pfx+n_kv]

    // cache-resident param sinks: refresh the reserved prefix rows (cheap, static values)
    build_sink_write(layer, k_row, il);

    ggml_tensor * k = idx_d == 0 ? k_row
        : ggml_view_4d(ctx0, k_row, kv_lora_rank + n_embd_head_qk_rope, k_row->ne[1], k_row->ne[2], k_row->ne[3],
                k_row->nb[1], k_row->nb[2], k_row->nb[3], 0);             // MLA part: [kv_lora+rope, 1, n_pfx+n_kv]

    ggml_tensor * v = ggml_view_4d(ctx0, k_row, kv_lora_rank, k_row->ne[1], k_row->ne[2], k_row->ne[3],
            k_row->nb[1], k_row->nb[2], k_row->nb[3], 0);

    // hoisted combined mask [n_pfx | n_kv]; DSA layers refine the cell part below
    ggml_tensor * kq_mask_all = is_swa ? kq_mask_sinked_swa : kq_mask_sinked;
    ggml_tensor * kq_mask     = is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask();

    // while the whole cache fits in the top-k budget the selection is a no-op:
    // skip the scoring (the indexer keys are still written to the cache above)
    if (is_dsa && k_row->ne[2] - n_pfx > (int64_t) hparams.indexer_top_k) {
        // score the cached indexer keys (cell rows only; the param sinks are not
        // scored -- they stay always-visible through the mask prefix):
        //   score[t,s] = sum_h w[t,h] * relu(q[t,h,:] . k[s,:])   (no scaling, per the reference)
        ggml_tensor * ik_cache = ggml_view_4d(ctx0, k_row, idx_d, k_row->ne[1], k_row->ne[2] - n_pfx, k_row->ne[3],
                k_row->nb[1], k_row->nb[2], k_row->nb[3],
                n_pfx*k_row->nb[2] + ggml_row_size(k_row->type, kv_lora_rank + n_embd_head_qk_rope)); // [idx_d, 1, n_kv]

        ggml_tensor * iw = ggml_mul_mat(ctx0, layer.indexer_proj, cur);          // [idx_h, nt]

        ggml_tensor * iscore;
        if (use_fused_sinkhorn && nt <= OPV2_HC_BATCHED_NT_MAX) {
            // decode: one fused pass over the cached indexer keys (incl. relu,
            // head-weighting and the additive mask)
            ggml_tensor * ikv = ggml_view_2d(ctx0, k_row, idx_d, k_row->ne[2] - n_pfx, k_row->nb[2],
                    n_pfx*k_row->nb[2] + ggml_row_size(k_row->type, kv_lora_rank + n_embd_head_qk_rope));
            iscore = ggml_dsa_score(ctx0, ikv, indexer_q, iw, kq_mask);          // [n_kv, nt]
        } else {
            // prefill: tiled matmul pipeline
            // split the batch into streams if needed
            const int64_t n_stream = ik_cache->ne[3];
            indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
                    indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
            iw = ggml_view_4d(ctx0, iw, iw->ne[0], iw->ne[1]/n_stream, iw->ne[2], n_stream,
                    iw->nb[1], iw->nb[2]/n_stream, iw->nb[3]/n_stream, 0);

            indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);               // [idx_d, nt, idx_h]
            ik_cache  = ggml_permute(ctx0, ik_cache,  0, 2, 1, 3);               // [idx_d, n_kv, 1]

            ggml_tensor * ikq = ggml_mul_mat(ctx0, ik_cache, indexer_q);         // [n_kv, nt, idx_h]
            ikq = ggml_relu(ctx0, ikq);                                          // contiguous, no cont needed
            ikq = ggml_cont(ctx0, ggml_permute(ctx0, ikq, 1, 2, 0, 3));          // [idx_h, n_kv, nt]

            // weighted sum over the heads as a batched matvec
            ggml_tensor * iwv = ggml_view_4d(ctx0, iw, iw->ne[0], 1, iw->ne[1], iw->ne[3],
                    iw->nb[2], iw->nb[1], iw->nb[3], 0);                         // [idx_h, 1, nt]
            iscore = ggml_mul_mat(ctx0, ikq, iwv);                               // [n_kv, 1, nt]
            iscore = ggml_reshape_4d(ctx0, iscore, iscore->ne[0], iscore->ne[2], 1, iscore->ne[3]); // [n_kv, nt, 1]

            // mask before top-k so invalid/future cells cannot claim slots
            iscore = ggml_add(ctx0, iscore,
                    kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32));
        }

        const int64_t n_top_k = std::min<int64_t>(iscore->ne[0], hparams.indexer_top_k);
        ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, iscore, n_top_k));

        if (nt == 1) {
            // gather-based decode attention: physically gather the selected rows and
            // attend over [sinks | top-k] only -- attention cost stays constant with
            // context instead of reading (and masking) the entire cache
            ggml_tensor * ids = ggml_reshape_1d(ctx0, top_k, n_top_k);

            ggml_tensor * cells = ggml_view_2d(ctx0, k_row, k_row->ne[0], k_row->ne[2] - n_pfx,
                    k_row->nb[2], n_pfx*k_row->nb[2]);                       // [row_w, n_kv]
            ggml_tensor * gk = ggml_get_rows(ctx0, cells, ids);              // [row_w, n_top_k] F32
            gk = ggml_cast(ctx0, gk, k_row->type);

            ggml_tensor * prefix = ggml_view_2d(ctx0, k_row, k_row->ne[0], n_pfx, k_row->nb[2], 0);
            ggml_tensor * kg = ggml_concat(ctx0, prefix, gk, 1);             // [row_w, n_pfx+n_top_k]

            const int64_t n_rows = n_pfx + n_top_k;
            k = ggml_view_4d(ctx0, kg, kv_lora_rank + n_embd_head_qk_rope, 1, n_rows, 1,
                    kg->nb[1], kg->nb[1], kg->nb[1]*n_rows, 0);
            v = ggml_view_4d(ctx0, kg, kv_lora_rank, 1, n_rows, 1,
                    kg->nb[1], kg->nb[1], kg->nb[1]*n_rows, 0);

            // gather the mask entries of the selected cells (keeps padding/other-seq
            // cells that sneak into the top-k masked out)
            ggml_tensor * m_r = ggml_reshape_2d(ctx0, kq_mask, 1, kq_mask->ne[0]);
            ggml_tensor * gm  = ggml_get_rows(ctx0, m_r, ids);               // [1, n_top_k] F32
            gm = ggml_reshape_2d(ctx0, gm, n_top_k, 1);
            if (gm->type != sink_mask_blk->type) {
                gm = ggml_cast(ctx0, gm, sink_mask_blk->type);
            }
            kq_mask_all = ggml_concat(ctx0, sink_mask_blk, gm, 0);           // [n_pfx+n_top_k, 1]
        } else {
            // prefill: every query has its own top-k set, so keep the mask-based form --
            // unmask the selected cells on an all-masked copy, then AND with the causal
            // mask (same construction as the DSA build_attn overload in llama-graph.cpp)
            ggml_tensor * mask_inf = ggml_fill(ctx0, kq_mask, -INFINITY);
            mask_inf = ggml_view_4d(ctx0, mask_inf, 1, mask_inf->ne[0], mask_inf->ne[1], mask_inf->ne[3],
                    mask_inf->nb[0], mask_inf->nb[1], mask_inf->nb[2], 0);
            ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
                    top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);
            ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
            zeros = ggml_fill(ctx0, zeros, 0.0f);
            ggml_tensor * unmasked = ggml_set_rows(ctx0, mask_inf, zeros, top_k_3d);
            unmasked = ggml_view_4d(ctx0, unmasked, unmasked->ne[1], unmasked->ne[2], 1, unmasked->ne[3],
                    unmasked->nb[2], unmasked->nb[3], unmasked->nb[3], 0);
            kq_mask = ggml_add(ctx0, unmasked, kq_mask);

            // per-DSA-layer combined mask (the top-k selection differs per layer)
            kq_mask_all = ggml_concat(ctx0, sink_mask_blk, kq_mask, 0);
        }
    }

    ggml_tensor * attn_out = build_attn_mha(Qcur, k, v, nullptr, kq_mask_all, nullptr, layer.wv_b, kq_scale, il);

    attn_out = build_mome_conv(attn_out, layer.attn_o_conv, conv_rs, conv_state, rs_head, off_o, n_seqs, n_seq_tokens);
    attn_out = ggml_mul_mat(ctx0, layer.wo, attn_out);                  // o_proj
    return attn_out;
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_attention_mtp(const llama_model & model, llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * cur, ggml_tensor * inp_pos, float kq_scale, int il) const {
    const auto & layer = model.layers[il];
    // in the combined GGUF the MTP layers are converted as full attention (the single
    // global n_swa cannot express the reference's 2048 next to the trunk's 512); a
    // split MTP-only GGUF carries its own n_swa = 2048 and runs SWA here, matching
    // the reference -- see conversion/pangu.py and conversion/split_pangu_mtp.py
    const bool is_swa = hparams.is_swa(il);

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t nt                  = cur->ne[1];

    // q down -> norm -> up (no MoME conv on the MTP head)
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_mul_mat(ctx0, layer.wq_b, q);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, nt,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, nt,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    kv_cmpr = ggml_cont(ctx0, kv_cmpr);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_qk_rope, GGML_ROPE_TYPE_NEOX,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_embd_head_qk_rope, GGML_ROPE_TYPE_NEOX,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, nt);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);

    // MTP layers have no indexer -- zero-pad the DSA tail of the cache row
    const int64_t idx_d = hparams.indexer_head_size;
    if (idx_d > 0) {
        Kcur = ggml_pad(ctx0, Kcur, idx_d, 0, 0, 0);
    }

    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Kcur);
    const auto * attn_mctx = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
    ggml_tensor * k_idxs = is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
    ggml_build_forward_expand(gf, attn_mctx->cpy_k(ctx0, Kcur, k_idxs, il));

    ggml_tensor * k_row = attn_mctx->get_k(ctx0, il);                   // [row_w, 1, n_pfx+n_kv]

    // cache-resident param sinks: refresh the reserved prefix rows
    build_sink_write(layer, k_row, il);

    ggml_tensor * k = idx_d == 0 ? k_row
        : ggml_view_4d(ctx0, k_row, kv_lora_rank + n_embd_head_qk_rope, k_row->ne[1], k_row->ne[2], k_row->ne[3],
                k_row->nb[1], k_row->nb[2], k_row->nb[3], 0);

    ggml_tensor * v = ggml_view_4d(ctx0, k_row, kv_lora_rank, k_row->ne[1], k_row->ne[2], k_row->ne[3],
            k_row->nb[1], k_row->nb[2], k_row->nb[3], 0);

    // hoisted combined mask
    ggml_tensor * kq_mask_all = is_swa ? kq_mask_sinked_swa : kq_mask_sinked;

    ggml_tensor * attn_out = build_attn_mha(Qcur, k, v, nullptr, kq_mask_all, nullptr, layer.wv_b, kq_scale, il);
    attn_out = ggml_mul_mat(ctx0, layer.wo, attn_out);   // o_proj (no MoME conv)
    return attn_out;
}

ggml_tensor * llama_model_openpangu_v2::graph_base::build_moe_block(const llama_model & model, ggml_tensor * cur, int il) const {
    if (il < (int) hparams.n_layer_dense_lead) {
        return build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    ggml_tensor * moe_out = build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            model.layers[il].ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    ggml_tensor * ffn_shexp = build_ffn(cur,
            model.layers[il].ffn_up_shexp,   nullptr, nullptr,
            model.layers[il].ffn_gate_shexp, nullptr, nullptr,
            model.layers[il].ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    return ggml_add(ctx0, moe_out, ffn_shexp);
}

std::unique_ptr<llm_graph_context> llama_model_openpangu_v2::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_openpangu_v2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    graph_base(params, static_cast<const llama_model_openpangu_v2 &>(model).sinkhorn_fused()) {
    GGML_ASSERT(n_layer > 0 && "openpangu-v2: no trunk layers -- a split MTP-only GGUF can only be used as --model-draft with --mtp");

    const int64_t hc = hparams.dsv4_hc_mult;
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_mla()));

    ggml_tensor * cur;
    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();

    // hoisted sink masks: one [n_pfx | n_kv] concat per variant per graph
    {
        ggml_tensor * mask_base = inp_hybrid->get_attn()->get_kq_mask();
        ggml_tensor * mask_swa  = inp_hybrid->get_attn()->get_kq_mask_swa();
        sink_mask_blk      = build_sink_mask_blk(mask_base);
        kq_mask_sinked     = ggml_concat(ctx0, sink_mask_blk, mask_base, 0);
        kq_mask_sinked_swa = ggml_concat(ctx0, build_sink_mask_blk(mask_swa), mask_swa, 0);
    }

    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);

    // transposed stream state [hc, n_embd, nt] for the batched-matmul mHC form,
    // shared by the pre- and post-merge of a block; null selects the loop form
    auto make_xt = [&](ggml_tensor * s) -> ggml_tensor * {
        return s->ne[2] <= OPV2_HC_BATCHED_NT_MAX
            ? ggml_cont(ctx0, ggml_permute(ctx0, s, 1, 0, 2, 3))
            : nullptr;
    };

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        // attention block
        ggml_tensor * residual   = inpL;
        ggml_tensor * residual_t = make_xt(inpL);
        cur = build_hc_pre(inpL, residual_t, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base, &post, &comb, il);
        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_attention(model, inp_hybrid, cur, inp_pos, kq_scale, il);
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        inpL = build_hc_post(cur, residual, residual_t, post, comb, il);

        // the last layer's FFN only needs the requested output rows -- gather early,
        // unless the unmasked MTP extraction needs the full-length hidden state
        if (il == n_layer - 1 && inp_out_ids &&
            (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
            flat = ggml_get_rows(ctx0, flat, inp_out_ids);
            inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
        }

        // feed-forward block
        residual   = inpL;
        residual_t = make_xt(inpL);
        cur = build_hc_pre(inpL, residual_t, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base, &post, &comb, il);
        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);

        cur = build_moe_block(model, cur, il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        inpL = build_hc_post(cur, residual, residual_t, post, comb, il);

        // block post norm on the stream-concatenated residual (subset of layers)
        if (model.layers[il].block_post_norm) {
            const int64_t nt_l = inpL->ne[2];
            ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, nt_l);
            flat = build_norm(flat, model.layers[il].block_post_norm, nullptr, LLM_NORM_RMS, il);
            inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, nt_l);
        }

        inpL = build_cvec(inpL, il);
    }

    // Collapse the mHC streams. In the unmasked-MTP case inpL still holds all
    // n_tokens rows so t_h_nextn covers every position; otherwise the last-layer
    // gather above already reduced it to the output rows.
    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);

    // trunk hidden state (pre-output-norm), consumed by the MTP head for speculative drafting
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_openpangu_v2::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph_base(params, static_cast<const llama_model_openpangu_v2 &>(model).sinkhorn_fused()) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "openpangu-v2 MTP requires n_layer_nextn > 0");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn);
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm);

    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_mla()));

    // inputs: the next token id + the previous hidden state (mtp_h_input)
    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * h_input   = inp->embd;
    ggml_tensor * embd_w    = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd  = ggml_get_rows(ctx0, embd_w, inp->tokens);
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();

    // hoisted sink masks (the swa variant is live for split MTP-only models, where
    // the MTP layers run their reference SWA-2048 window)
    {
        ggml_tensor * mask_base = inp_attn->get_kq_mask();
        ggml_tensor * mask_swa  = inp_attn->get_kq_mask_swa();
        sink_mask_blk      = build_sink_mask_blk(mask_base);
        kq_mask_sinked     = ggml_concat(ctx0, sink_mask_blk, mask_base, 0);
        kq_mask_sinked_swa = ggml_concat(ctx0, build_sink_mask_blk(mask_swa), mask_swa, 0);
    }

    // eh_proj( concat[ enorm(embed(next_tok)), hnorm(prev_hidden) ] )
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * h_norm = build_norm(h_input,  layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));

    // plain (non-mHC) decoder layer: sandwich norms + MLA(+sinks, no conv) + MoE
    ggml_tensor * res_sa = cur;
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cur = build_attention_mtp(model, inp_attn, cur, inp_pos, kq_scale, il);
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cur = ggml_add(ctx0, cur, res_sa);

    ggml_tensor * res_ffn = cur;
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cur = build_moe_block(model, cur, il);
    cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
    cur = ggml_add(ctx0, cur, res_ffn);

    // hidden state to seed the next chained MTP head: keep all n_tokens rows,
    // the unmasked nextn extraction reads them densely by token position
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    ggml_tensor * head_norm = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    ggml_tensor * head_w    = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    cur = build_norm(cur, head_norm, nullptr, LLM_NORM_RMS, -1);
    cur = build_lora_mm(head_w, cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
