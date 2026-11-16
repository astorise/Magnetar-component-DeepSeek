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

**Implemented for the DeepSeekMoE decoder shape; MLA is not supported.**
`src/lib.rs` builds prefill, decode and layer-segment Execution Graphs
through `magnetar:model-component-graph@1.6.0` (vendored under `wit/`),
reading every dimension from `model-config` on each call.

Supported:

- pre-norm decoder layers, multi-head or grouped-query attention, half-split
  RoPE, no projection bias, optional tied embeddings;
- a dense gated MLP, or DeepSeekMoE: routed experts (`moe-experts`, softmax
  top-k, optional renormalisation) plus shared experts as an *ungated* gated
  MLP added to the routed output (unlike Qwen's sigmoid-gated one).

Refused with an explicit error rather than approximated (`unsupported_reason`):
attention bias, per-frequency RoPE scaling (YaRN), per-layer geometry, final
logit softcapping, per-layer input embeddings, inconsistent head or expert
counts.

Not expressible through the contract today, so not implemented:

- **Multi-head Latent Attention** (DeepSeek-V2/V3): the contract has a single
  `head-dimension`, no latent KV state, no decoupled RoPE key and no
  multi-output nodes, and `model-config` carries none of `kv_lora_rank`,
  `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`.
- **Dense leading layers** (`first_k_dense_replace`): `mixture-of-experts`
  applies to every layer, so a checkpoint mixing dense and MoE layers (such as
  DeepSeek-MoE-16B) cannot be described faithfully yet.
- V3 sigmoid routing with correction bias, group-limited routing,
  `routed_scaling_factor`, multi-token prediction and FP8 weights.

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
