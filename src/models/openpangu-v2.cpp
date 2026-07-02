#include "models.h"

#include "llama-kv-cache.h"

// openPangu-2.0-Flash (openpangu-v2)
//
// Composition of existing infra:
//   - 4-stream mHC residual (manifold hyper-connections), same family as DeepSeek-V4.
//     The learned RMSNorm gamma is folded into the phi weight at conversion time.
//   - MLA attention (absorbed), same as DeepSeek-V2, scale 1/sqrt(qk_head_dim).
//   - MoME: width-k causal depthwise conv + identity residual on q_a, kv-compressed, and attn-out.
//   - 128 learned param-sink KV entries per layer (always visible) -- TODO(stage B).
//   - Sigmoid-gated MoE with expert bias + shared expert, leading dense layers.
//
// This first version runs every layer as full causal attention (exact for prompts up to
// the DSA top-k / SWA window), and computes the MoME conv per batch with zero left-padding
// (exact within a single prefill). Cross-batch conv state and DSA/SWA masking are follow-ups.

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

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl, false);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    // mHC (reuse the DeepSeek-V4 hyper-connection fields)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    ml.get_key(LLM_KV_ATTENTION_SINK_COUNT,       hparams.openpangu_n_sink, false);
    ml.get_key(LLM_KV_ATTENTION_CONV_KERNEL_SIZE, hparams.openpangu_conv_k, false);

    // DSA lightning indexer (loaded but unused in this dense-attention version)
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k, false);

    // MLA stores a single compressed KV head: size the KV cache to the
    // compressed dims (kv_lora + rope for K, kv_lora for V). The real
    // per-head attention dims come from n_embd_head_{k,v}_mla().
    hparams.n_embd_head_k_full = hparams.n_lora_kv + hparams.n_rot();
    hparams.n_embd_head_v_full = hparams.n_lora_kv;

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

    // model-level mHC stream merge (pre_only)
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc * n_embd, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    const int64_t hc_mix_dim = (2 + hc) * hc;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, 0);
        layer.block_post_norm = create_tensor(tn(LLM_TENSOR_BLOCK_POST_NORM, "weight", i), {hc * n_embd}, TENSOR_NOT_REQUIRED);

        // mHC per-layer
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc * n_embd, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc * n_embd, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

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

        // DSA lightning indexer (present only on DSA layers; loaded, not used here)
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

ggml_tensor * llama_model_openpangu_v2::graph::build_hc_weighted_sum(ggml_tensor * x, ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih * x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih * weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }
    return acc;
}

ggml_tensor * llama_model_openpangu_v2::graph::build_hc_sinkhorn(ggml_tensor * comb, int il) const {
    GGML_UNUSED(il);

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

ggml_tensor * llama_model_openpangu_v2::graph::build_hc_pre(ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale,
        ggml_tensor * hc_base, ggml_tensor ** post, ggml_tensor ** comb, int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc * n_embd;
    const int64_t nt         = x->ne[2];

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);

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

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_openpangu_v2::graph::build_hc_post(ggml_tensor * x, ggml_tensor * residual,
        ggml_tensor * post, ggml_tensor * comb, int il) const {
    GGML_UNUSED(il);
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

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

ggml_tensor * llama_model_openpangu_v2::graph::build_hc_head(ggml_tensor * x, ggml_tensor * hc_fn,
        ggml_tensor * hc_scale, ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);

    ggml_tensor * pre = opv2_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_openpangu_v2::graph::build_mome_conv(ggml_tensor * x, ggml_tensor * conv_w) const {
    // x: [dim, n_tokens]; conv_w: [k, dim] (bf16 in gguf, cast to f32 for ssm_conv)
    const int64_t dim = x->ne[0];
    const int64_t nt  = x->ne[1];
    const int64_t k   = conv_w->ne[0];

    ggml_tensor * w = ggml_cast(ctx0, conv_w, GGML_TYPE_F32);

    ggml_tensor * xt  = ggml_cont(ctx0, ggml_transpose(ctx0, x));        // [nt, dim]
    ggml_tensor * pad = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, k - 1, dim);
    pad = ggml_fill(ctx0, pad, 0.0f);                                    // [k-1, dim]
    ggml_tensor * sx = ggml_concat(ctx0, pad, xt, 0);                    // [nt+k-1, dim]
    sx = ggml_reshape_3d(ctx0, sx, nt + k - 1, dim, 1);

    ggml_tensor * conv = ggml_ssm_conv(ctx0, sx, w);                     // [dim, nt, 1]
    conv = ggml_reshape_2d(ctx0, conv, dim, nt);

    return ggml_add(ctx0, x, conv);                                     // identity residual
}

ggml_tensor * llama_model_openpangu_v2::graph::build_attention(const llama_model & model, llm_graph_input_attn_k * inp_attn,
        ggml_tensor * cur, ggml_tensor * inp_pos, float kq_scale, int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k      = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t nt                  = cur->ne[1];

    // q down -> conv -> norm -> up
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    q = build_mome_conv(q, layer.attn_q_a_conv);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_mul_mat(ctx0, layer.wq_b, q);

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
    kv_cmpr = build_mome_conv(kv_cmpr, layer.attn_kv_a_conv);
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

    // store compressed K to the cache (MLA: V is a view of K, decompressed by wv_b)
    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Kcur);
    const auto * mctx = inp_attn->mctx;
    ggml_build_forward_expand(gf, mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));

    ggml_tensor * k = mctx->get_k(ctx0, il);                            // [kv_lora+rope, 1, n_kv]

    // learned param sinks: 128 always-visible compressed KV entries (k_pe not RoPE'd)
    const int64_t n_sink = hparams.openpangu_n_sink;
    ggml_tensor * sink_in = ggml_cast(ctx0, layer.attn_sink_kv, GGML_TYPE_F32);
    ggml_tensor * sink_kv = build_norm(sink_in, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * sink_pe = ggml_cast(ctx0, layer.attn_sink_k_pe, sink_kv->type);
    ggml_tensor * sink_k  = ggml_concat(ctx0, sink_kv, sink_pe, 0);     // [kv_lora+rope, n_sink]
    sink_k = ggml_reshape_4d(ctx0, sink_k, kv_lora_rank + n_embd_head_qk_rope, 1, n_sink, 1);
    sink_k = ggml_cast(ctx0, sink_k, k->type);
    if (k->ne[3] != 1) {
        sink_k = ggml_repeat_4d(ctx0, sink_k, sink_k->ne[0], sink_k->ne[1], sink_k->ne[2], k->ne[3]);
    }
    k = ggml_concat(ctx0, sink_k, k, 2);                               // sinks first

    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3],
            k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    ggml_tensor * sink_mask = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sink,
            kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
    sink_mask = ggml_fill(ctx0, sink_mask, 0.0f);                       // always visible
    kq_mask = ggml_concat(ctx0, sink_mask, kq_mask, 0);

    ggml_tensor * attn_out = build_attn_mha(Qcur, k, v, nullptr, kq_mask, nullptr, layer.wv_b, kq_scale, il);

    attn_out = build_mome_conv(attn_out, layer.attn_o_conv);
    attn_out = ggml_mul_mat(ctx0, layer.wo, attn_out);                  // o_proj
    return attn_out;
}

std::unique_ptr<llm_graph_context> llama_model_openpangu_v2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_openpangu_v2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_mla()));

    ggml_tensor * cur;
    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        // attention block
        cur = build_hc_pre(inpL, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base, &post, &comb, il);
        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_attention(model, inp_attn, cur, inp_pos, kq_scale, il);
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        inpL = build_hc_post(cur, residual, post, comb, il);

        // feed-forward block
        residual = inpL;
        cur = build_hc_pre(inpL, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base, &post, &comb, il);
        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);

        if (il < (int) hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
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
            cur = ggml_add(ctx0, moe_out, ffn_shexp);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        inpL = build_hc_post(cur, residual, post, comb, il);

        // block post norm on the stream-concatenated residual (subset of layers)
        if (model.layers[il].block_post_norm) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
            flat = build_norm(flat, model.layers[il].block_post_norm, nullptr, LLM_NORM_RMS, il);
            inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_tokens);
        }

        inpL = build_cvec(inpL, il);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
