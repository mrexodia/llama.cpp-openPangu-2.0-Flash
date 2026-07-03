# llama.cpp — openPangu-2.0-Flash fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds full support for
[openPangu-2.0-Flash](https://huggingface.co/openpangu/openPangu-2.0-Flash)
(92B MoE, ~6B active parameters, 512K context). This architecture is not supported upstream.

**Get the GGUF models here: https://huggingface.co/mrexodia/openPangu-2.0-Flash-GGUF**

## What this fork adds

- The `openpangu-v2` architecture: absorbed MLA attention, DSA sparse attention
  (lightning-indexer top-2048 selection on the global layers), per-layer sliding-window
  attention via a hybrid iSWA cache, 4-stream manifold hyper-connections (mHC),
  MoME causal convolutions with recurrent cross-batch state, and 128 learned
  cache-resident attention sinks per layer
- Fused CUDA/CPU kernels for the mHC mixing chain (`GGML_OP_SINKHORN`,
  `GGML_OP_HC_MIX`) — decode is ~40% faster than the naive graph, with automatic
  fallback to the unfused ops on backends without the kernels (e.g. Metal)
- HF → GGUF conversion (`convert_hf_to_gguf.py`) and a base/MTP split tool
  (`conversion/split_pangu_mtp.py`)
- Multi-token-prediction self-speculative decoding (`--mtp`, with the split MTP draft GGUF)
- Chat support: tool calling and `<think>` reasoning parsing for the openPangu template
- A CUDA fix for non-contiguous `CONCAT` beyond 65535 rows (needed past 64K context)

## Quickstart

```sh
cmake -B build -DGGML_CUDA=ON       # Metal is enabled by default on Apple Silicon
cmake --build build -j --target llama-server

build/bin/llama-server -m openPangu-2.0-Flash-base-Q8_0.gguf -c 65536 --jinja
```

Context scales to `-c 524288`; the compressed MLA KV cache needs only ~12 GB at the full 512K.

## Measured performance

DGX Spark (GB10, unified memory):

| llama-bench | Q4_K_M | Q8_0 |
|---|---|---|
| pp512 | 770 t/s | 575 t/s |
| tg128 | 38.0 t/s | 23.7 t/s |

| llama-server, Q4_K_M | short | @10K | @24K | @100K |
|---|---|---|---|---|
| Prompt processing | n/a¹ | 525 t/s | 359 t/s | 134 t/s |
| Generation | ~25 t/s | 21.5 t/s | 18.4 t/s | 10.4 t/s |

¹ prompt-processing throughput is not meaningful for very short prompts (dominated by
fixed per-request overhead) — see `pp512` above.

Quality (perplexity on clean English prose, `-c 2048`): Q4_K_M **3.45**, Q3_K_M **3.70**.
Needle-in-a-haystack retrieval validated at 10K, 24K and 100K tokens. MTP drafting reaches
~91% acceptance with the split draft model (worthwhile mainly on discrete GPUs).

On a 64 GB Mac Studio use `Q3_K_M` (41.7 GB) and raise the Metal wired limit:

```sh
sudo sysctl iogpu.wired_limit_mb=57344
```

## Upstream

Everything else is unchanged [llama.cpp](https://github.com/ggml-org/llama.cpp) — see the
upstream repository for general documentation, the full tool list and platform support.
Licensed MIT, same as upstream.
