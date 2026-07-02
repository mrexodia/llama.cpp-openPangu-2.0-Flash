from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("PanguEmbeddedForCausalLM")
class PanguEmbeddedModel(TextModel):
    model_arch = gguf.MODEL_ARCH.PANGU_EMBED

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # PanguEmbedded's hparam loaded from config.json without head_dim
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        if hparams.get("head_dim") is None:
            self.gguf_writer.add_key_length(rope_dim)
            self.gguf_writer.add_value_length(rope_dim)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("OpenPanguV2ForCausalLM")
class OpenPanguV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.OPENPANGU_V2

    # simple root tensor map (HF name -> gguf enum)
    _root_map = {
        "model.embed_tokens.weight":               (gguf.MODEL_TENSOR.TOKEN_EMBD,    ".weight"),
        "model.norm.weight":                       (gguf.MODEL_TENSOR.OUTPUT_NORM,   ".weight"),
        "lm_head.weight":                          (gguf.MODEL_TENSOR.OUTPUT,        ".weight"),
        "model.merge_mhc_module.branch_alpha_pre": (gguf.MODEL_TENSOR.HC_HEAD_SCALE, ".weight"),
        "model.merge_mhc_module.branch_beta_pre":  (gguf.MODEL_TENSOR.HC_HEAD_BASE,  ".weight"),
    }

    # per-layer tensor map (suffix after "model.layers.N." -> gguf enum)
    _layer_map = {
        "input_layernorm.weight":                  (gguf.MODEL_TENSOR.ATTN_NORM,       ".weight"),
        "post_attention_layernorm.weight":         (gguf.MODEL_TENSOR.ATTN_POST_NORM,  ".weight"),
        "pre_mlp_layernorm.weight":                (gguf.MODEL_TENSOR.FFN_NORM,        ".weight"),
        "post_mlp_layernorm.weight":               (gguf.MODEL_TENSOR.FFN_POST_NORM,   ".weight"),
        "block_post_layernorm.weight":             (gguf.MODEL_TENSOR.BLOCK_POST_NORM, ".weight"),
        "attn_mhc_module.branch_alpha":            (gguf.MODEL_TENSOR.HC_ATTN_SCALE,   ".weight"),
        "attn_mhc_module.branch_beta":             (gguf.MODEL_TENSOR.HC_ATTN_BASE,    ".weight"),
        "mlp_mhc_module.branch_alpha":             (gguf.MODEL_TENSOR.HC_FFN_SCALE,    ".weight"),
        "mlp_mhc_module.branch_beta":              (gguf.MODEL_TENSOR.HC_FFN_BASE,     ".weight"),
        "self_attn.q_a_proj.weight":               (gguf.MODEL_TENSOR.ATTN_Q_A,        ".weight"),
        "self_attn.q_a_layernorm.weight":          (gguf.MODEL_TENSOR.ATTN_Q_A_NORM,   ".weight"),
        "self_attn.q_b_proj.weight":               (gguf.MODEL_TENSOR.ATTN_Q_B,        ".weight"),
        "self_attn.qa_conv.weight":                (gguf.MODEL_TENSOR.ATTN_Q_A_CONV,   ".weight"),
        "self_attn.kv_a_proj_with_mqa.weight":     (gguf.MODEL_TENSOR.ATTN_KV_A_MQA,   ".weight"),
        "self_attn.kv_a_layernorm.weight":         (gguf.MODEL_TENSOR.ATTN_KV_A_NORM,  ".weight"),
        "self_attn.compresskv_conv.weight":        (gguf.MODEL_TENSOR.ATTN_KV_A_CONV,  ".weight"),
        "self_attn.o_conv.weight":                 (gguf.MODEL_TENSOR.ATTN_O_CONV,     ".weight"),
        "self_attn.o_proj.weight":                 (gguf.MODEL_TENSOR.ATTN_OUT,        ".weight"),
        "self_attn.param_sink_compressed_kv":      (gguf.MODEL_TENSOR.ATTN_SINK_KV,    ".weight"),
        "self_attn.param_sink_k_pe":               (gguf.MODEL_TENSOR.ATTN_SINK_K_PE,  ".weight"),
        "self_attn.indexer.wq_b.weight":           (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
        "self_attn.indexer.wk.weight":             (gguf.MODEL_TENSOR.INDEXER_ATTN_K,  ".weight"),
        "self_attn.indexer.k_norm.weight":         (gguf.MODEL_TENSOR.INDEXER_K_NORM,  ".weight"),
        "self_attn.indexer.weights_proj.weight":   (gguf.MODEL_TENSOR.INDEXER_PROJ,    ".weight"),
        "mlp.gate.weight":                         (gguf.MODEL_TENSOR.FFN_GATE_INP,    ".weight"),
        "mlp.e_score_correction.bias":             (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
        "mlp.shared_experts.gate_proj.weight":     (gguf.MODEL_TENSOR.FFN_GATE_SHEXP,  ".weight"),
        "mlp.shared_experts.up_proj.weight":       (gguf.MODEL_TENSOR.FFN_UP_SHEXP,    ".weight"),
        "mlp.shared_experts.down_proj.weight":     (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP,  ".weight"),
        "mlp.gate_proj.weight":                    (gguf.MODEL_TENSOR.FFN_GATE,        ".weight"),
        "mlp.up_proj.weight":                      (gguf.MODEL_TENSOR.FFN_UP,          ".weight"),
        "mlp.down_proj.weight":                    (gguf.MODEL_TENSOR.FFN_DOWN,        ".weight"),
        # MTP / NextN heads (layers >= num_hidden_layers)
        "enorm.weight":                            (gguf.MODEL_TENSOR.NEXTN_ENORM,            ".weight"),
        "hnorm.weight":                            (gguf.MODEL_TENSOR.NEXTN_HNORM,            ".weight"),
        "eh_proj.weight":                          (gguf.MODEL_TENSOR.NEXTN_EH_PROJ,          ".weight"),
        "embed_tokens.weight":                     (gguf.MODEL_TENSOR.NEXTN_EMBED_TOKENS,     ".weight"),
        "shared_head.norm.weight":                 (gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM, ".weight"),
        "shared_head.head.weight":                 (gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD, ".weight"),
    }

    _expert_map = {
        "gate_proj": gguf.MODEL_TENSOR.FFN_GATE_EXP,
        "up_proj":   gguf.MODEL_TENSOR.FFN_UP_EXP,
        "down_proj": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # buffers for mHC gamma-fold and expert stacking
        self._mhc_buf: dict = {}
        self._exp_buf: dict = {}
        # include the MTP / NextN prediction layers in the block count
        self._n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        if self._n_nextn > 0:
            self.block_count = self.hparams["num_hidden_layers"] + self._n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # tiny depthwise conv kernels, consumed as F32 by ggml_ssm_conv
        if new_name.endswith("_conv.weight") and ".attn_" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        # openPangu ships a custom tokenizer_class in tokenizer_config.json, but
        # tokenizer.json is a standard ByteLevel BPE. Allow AutoTokenizer to load
        # it without prompting for the custom-code confirmation.
        from transformers import AutoTokenizer
        orig = AutoTokenizer.from_pretrained

        def patched(*args, **kwargs):
            kwargs.setdefault("trust_remote_code", True)
            return orig(*args, **kwargs)

        AutoTokenizer.from_pretrained = staticmethod(patched)
        try:
            self._set_vocab_gpt2()
        finally:
            AutoTokenizer.from_pretrained = orig

    def set_gguf_parameters(self):
        hparams = self.hparams
        # MLA uses a single compressed KV head
        hparams.setdefault("num_key_value_heads", 1)

        super().set_gguf_parameters()

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])

        self.gguf_writer.add_hyper_connection_count(hparams["mhc_num_stream"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["mhc_recur_norm"])
        self.gguf_writer.add_hyper_connection_epsilon(1e-6)

        self.gguf_writer.add_sink_count(hparams["param_sink_number"])
        self.gguf_writer.add_conv_kernel_size(hparams["router_sliding_window"])

        if self._n_nextn > 0:
            self.gguf_writer.add_nextn_predict_layers(self._n_nextn)

        # per-layer sliding window: DSA layers get 0, SWA layers get their window.
        # NOTE: the reference runs the MTP / NextN layers with SWA (window 2048,
        # sliding_window_list[-1]), but llama.cpp has a single global window size and
        # the trunk SWA layers use 512, so the MTP layers are converted as global (0).
        # This diverges from the reference at contexts beyond the MTP window and can
        # only lower draft acceptance (drafts are verified by the trunk).
        n_layer = hparams["num_hidden_layers"]
        swa_layers = hparams["swa_layers"]
        window_list = hparams["sliding_window_list"]
        layer_window = []
        for il in range(self.block_count):
            if il < n_layer and il in swa_layers:
                layer_window.append(int(window_list[swa_layers.index(il)]))
            else:
                layer_window.append(0)
        self.gguf_writer.add_sliding_window(int(hparams["sliding_window"]))
        self.gguf_writer.add_layer_swa_window(layer_window)

    def _fold_mhc(self, key, tensor_enum, bid, data_torch, is_phi):
        buf = self._mhc_buf.setdefault(key, {})
        buf["phi" if is_phi else "gamma"] = data_torch
        if "phi" not in buf or "gamma" not in buf:
            return []
        # fold RMSNorm gamma into phi: linear(x*gamma, W) == linear(x, W*gamma)
        folded = (buf["phi"].float() * buf["gamma"].float())
        self._mhc_buf.pop(key)
        return [(self.format_tensor_name(tensor_enum, bid), folded)]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # MLA absorption: split kv_b_proj into k_b (transposed) and v_b
        if bid is not None and name.endswith("self_attn.kv_b_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope = self.hparams["qk_nope_head_dim"]
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2).contiguous()
            v_b = v_b.contiguous()
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid), k_b),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid), v_b),
            ]

        # mHC phi + norm_gamma get folded together
        if name == "model.merge_mhc_module.phi.weight":
            return self._fold_mhc("merge", gguf.MODEL_TENSOR.HC_HEAD_FN, None, data_torch, True)
        if name == "model.merge_mhc_module.norm_gamma":
            return self._fold_mhc("merge", gguf.MODEL_TENSOR.HC_HEAD_FN, None, data_torch, False)
        if name in self._root_map:
            enum, suffix = self._root_map[name]
            return [(self.format_tensor_name(enum, None, suffix), data_torch)]

        if bid is not None:
            suffix = name.split(f"model.layers.{bid}.", 1)[-1]

            for mod, fn_enum in (("attn_mhc_module", gguf.MODEL_TENSOR.HC_ATTN_FN),
                                 ("mlp_mhc_module",  gguf.MODEL_TENSOR.HC_FFN_FN)):
                if suffix == f"{mod}.phi.weight":
                    return self._fold_mhc((bid, mod), fn_enum, bid, data_torch, True)
                if suffix == f"{mod}.norm_gamma":
                    return self._fold_mhc((bid, mod), fn_enum, bid, data_torch, False)

            # routed experts: stack across the expert dimension
            if ".mlp.experts." in name:
                parts = suffix.split(".")  # mlp experts E proj weight
                eid = int(parts[2])
                proj = parts[3]
                ebuf = self._exp_buf.setdefault((bid, proj), {})
                ebuf[eid] = data_torch
                n_experts = self.hparams["n_routed_experts"]
                if len(ebuf) < n_experts:
                    return []
                stacked = torch.stack([ebuf[i] for i in range(n_experts)], dim=0)
                self._exp_buf.pop((bid, proj))
                return [(self.format_tensor_name(self._expert_map[proj], bid), stacked)]

            if suffix in self._layer_map:
                enum, sfx = self._layer_map[suffix]
                out = data_torch
                # depthwise conv weights are [C, 1, k]; drop the singleton group dim.
                # store as F32 so ggml_ssm_conv uses them without a per-eval cast
                # (like Mamba's ssm_conv1d; they are tiny)
                if enum in (gguf.MODEL_TENSOR.ATTN_Q_A_CONV,
                            gguf.MODEL_TENSOR.ATTN_KV_A_CONV,
                            gguf.MODEL_TENSOR.ATTN_O_CONV):
                    if out.ndim == 3:
                        out = out.squeeze(1)
                    out = out.float()
                return [(self.format_tensor_name(enum, bid, sfx), out)]

        logger.warning(f"openpangu-v2: unmapped tensor '{name}'")
        return []
