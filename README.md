# magnetar-component-deepseek

## Purpose

A WebAssembly Component implementing DeepSeek and compatible GLM latent-attention graphs for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It will
supply prefill/decode graph semantics for DeepSeek-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

**Implemented against `magnetar:model-component-graph@1.10.0`** (vendored under
`wit/`). `src/lib.rs` builds prefill, decode and layer-segment Execution
Graphs from `model-config` alone, and reads every `1.7.0` function on each
build (the Runtime refuses a graph for a checkpoint whose declared feature
was never read). It requires the checkpoint activation to be known and equal
to `silu`.

Supported:

- pre-norm decoder layers, no projection bias, optional tied embeddings;
- ordinary multi/grouped-query attention with half-split RoPE (DeepSeekMoE);
- **Multi-head Latent Attention** (DeepSeek-V2/V2-Lite), *expanded* form:
  split projections (`q_a_proj`/`q_a_norm` when `q_lora_rank` is set,
  `q_nope_proj`, `q_rope_proj`, `kv_a_proj`, `kv_a_norm`, `k_rope_proj`,
  `k_nope_proj`, `v_proj`), interleaved RoPE on the rotated parts, head-wise
  `concat` to `[nope | rope]` for Q and K (the shared rotated key given to
  every head), with the checkpoint's declared widths. Prefill expands the
  latent cache and decode uses absorbed query/key and value projections;
- **YaRN** RoPE on every `rope` node, and its softmax rescaling
  `scale = head_dim^-0.5 * mscale(factor, mscale_all_dim)^2`;
- per layer, a dense gated MLP or mixture of experts
  (`mixture-of-experts-layers`): softmax routing, or selection-biased sigmoid
  routing with optional top-two group scoring and explicit normalization.
  The routed output is scaled and shared experts use an ungated gated MLP;
- a separate epsilon for the query/KV bottleneck norms when declared.

Refused with an explicit error rather than approximated (`unsupported_reason`):
sigmoid routing without its bias or explicit normalization, other grouped
scoring modes, a gated shared expert, banded RoPE scaling, a linear scaling
factor, attention bias, per-layer geometry, final logit softcapping,
per-layer input embeddings, and inconsistent head, expert or layer counts.

In the Magnetar checkout, tiny synthetic `deepseek_v2` and `glm4_moe_lite`
bundles are compared with Transformers on CPU for prefill and cached decode.
GLM's top-1 and grouped top-2 routing are tested separately. The full GLM
checkpoint and CUDA sigmoid routing have not been qualified.

The DeepSeek-R1 distilled checkpoints are Qwen2.5 and Llama models and run
through `components/qwen` and `components/llama`, not this Component.

## Governing contract

This Component implements the `magnetar:model-component-graph` world exactly
as `components/qwen` and `components/llama` do. Family-specific requirements
are scoped in the main repository's `openspec/`
(`qualify-additional-model-and-state-families`).

## Relationship to magnetar-runtime

This Component is loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never runs with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/deepseek`.

## Continuous integration

`.github/workflows/ci.yml` calls the Module CI workflow the main Magnetar
repository defines once for all module repositories (formatting, Clippy,
tests with the committed `Cargo.lock`, `cargo-deny`), building this crate at
`components/deepseek` inside a Magnetar checkout.
