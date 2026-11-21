# magnetar-component-deepseek

## Purpose

A WebAssembly Component implementing the DeepSeek model families for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It will
supply prefill/decode graph semantics for DeepSeek-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

**Implemented against `magnetar:model-component-graph@1.8.0`** (vendored under
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
  every head), `attention` with `head_dimension` 192 and
  `value_head_dimension` 128. K and V are cached in full; the latent cache is
  a later Runtime milestone;
- **YaRN** RoPE on every `rope` node, and its softmax rescaling
  `scale = head_dim^-0.5 * mscale(factor, mscale_all_dim)^2`;
- per layer, a dense gated MLP or DeepSeekMoE (`mixture-of-experts-layers`):
  routed softmax top-k experts, scaled by `routed-scaling-factor`, plus shared
  experts as an *ungated* gated MLP.

Refused with an explicit error rather than approximated (`unsupported_reason`):
sigmoid, group-limited or biased expert routing (DeepSeek-V3, Runtime
milestone 3), a gated shared expert, banded RoPE scaling, a linear scaling
factor, attention bias, per-layer geometry, final logit softcapping,
per-layer input embeddings, and inconsistent head, expert or layer counts.

Not verified end to end: the graphs are unit-tested for their decisions only.
The acceptance runs (tiny random `deepseek_v2` bundles against transformers on
the Reference CPU, then DeepSeek-V2-Lite from a quantized GGUF on CUDA) need
this Component packaged into a Magnetar checkout. See
[astorise/Magnetar#198](https://github.com/astorise/Magnetar/issues/198).

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
