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

**Empty template.** This crate is a scaffold: no DeepSeek graph logic, no WIT
bindings and no Component Model exports exist here yet, and the Runtime has
no DeepSeek implementation of its own either. It is pinned into Magnetar as a
placeholder so the work has a home; it is not yet scheduled against an
OpenSpec change of its own.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- DeepSeek-V2/V3 use Multi-head Latent Attention: keys and values are reconstructed from a compressed per-token latent, so the KV cache stores that latent (plus a decoupled RoPE key) rather than per-head keys and values. Neither the attention Operator nor the KV layout models this today.
- DeepSeekMoE: shared experts plus many fine-grained routed experts per layer, with data-dependent top-k routing that must stay on the Device for step replay to keep working (`execute-and-cache-moe-experts`).
- DeepSeek-V3 adds multi-token prediction heads (`verify-speculative-and-mtp-generation`) and ships FP8 weights (`explore-checkpoint-native-low-precision`).
- The DeepSeek-R1 distilled checkpoints are Qwen2.5 and Llama models; they already run through `components/qwen` and `components/llama`, not this Component.

## Governing contract

None dedicated yet. A real implementation will target the same
`model-component-graph-contract` WIT interface (`magnetar:model-component-graph`)
`components/qwen` and `components/llama` implement, extended additively
where this family needs Operators or state the contract does not model yet.
Family-specific requirements will be scoped in the main repository's
`openspec/` (the `qualify-additional-model-and-state-families` change tracks
qualifying new families).

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/deepseek`.
