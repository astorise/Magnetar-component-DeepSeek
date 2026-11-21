//! DeepSeek Model Component for the Magnetar local AI Runtime
//! (`magnetar:model-component-graph@1.7.0`).
//!
//! Builds prefill and decode Execution Graphs for the DeepSeek decoder
//! families, from `model-config` alone:
//!
//! - **DeepSeekMoE** (`model_type = "deepseek"`): multi-head or grouped-query
//!   attention with half-split RoPE;
//! - **DeepSeek-V2/V3 attention**: Multi-head Latent Attention over a latent
//!   KV history (`cache-latent-attention-history`): the prefill attends in the
//!   expanded form over the history's rows (`chunk-latent-prefills`), a decode
//!   step in the absorbed form; with YaRN RoPE and its softmax rescaling;
//! - a feed-forward that is, per layer, a dense gated MLP or DeepSeekMoE:
//!   routed experts plus an always-active, **ungated** shared expert.
//!
//! The Runtime refuses a graph built without reading a `1.7.0` function whose
//! feature the checkpoint declares, so every build reads all of them. What
//! this Component does not apply it refuses to build rather than ignores (see
//! [`unsupported_reason`]): sigmoid or group-limited routing, selection bias
//! (DeepSeek-V3, Runtime milestone 3), banded RoPE scaling, a linear scaling
//! factor, per-layer geometry, softcapping and per-layer inputs.
//!
//! DeepSeek-R1 *distilled* checkpoints are Qwen2.5 and Llama models and run
//! through those Components, not this one.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape, add_node,
    alias_weight_edge, begin_graph, declare_input, finish_graph, kv_resource, weight_edge,
};
use magnetar::model_component_graph::model_config::{
    self, ArchitectureConfig, ExpertRoutingConfig, ExpertScoring, MlaConfig, MoeConfig,
    RopeFrequencyScaling, YarnRopeScaling,
};

/// The architecture family this Component serves.
pub const ARCHITECTURE_FAMILY: &str = "deepseek";

const ROPE_POSITION_MODE: &str = "sequential";

// ---------------------------------------------------------------------------
// What the bound configuration declares, and whether it can be built.
// ---------------------------------------------------------------------------

/// The RoPE frequency scaling a checkpoint declares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScalingFacts {
    None,
    Banded,
    Yarn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaFacts {
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u64,
    pub qk_rope_head_dim: u64,
    pub v_head_dim: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeFacts {
    pub experts: u32,
    pub experts_per_token: u32,
    pub has_shared_expert: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutingFacts {
    pub sigmoid: bool,
    pub has_groups: bool,
    pub routed_scaling_factor: f64,
    pub selection_bias: bool,
    pub shared_expert_gated: bool,
}

/// The facts about a bound configuration that decide whether this Component
/// can build a faithful graph. Plain data, so the decision is unit-testable
/// without a Runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportFacts {
    pub attention_bias: bool,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dimension: u64,
    /// `architecture-config.rope-scaling-factor`: a linear scaling factor.
    pub rope_scaling_factor: Option<f64>,
    pub rope_scaling: RopeScalingFacts,
    pub has_layer_layout: bool,
    pub has_final_logit_softcapping: bool,
    pub has_per_layer_input: bool,
    pub mla: Option<MlaFacts>,
    pub moe: Option<MoeFacts>,
    /// `mixture-of-experts-layers`, one entry per layer when declared.
    pub moe_layers: Option<Vec<bool>>,
    pub routing: Option<RoutingFacts>,
}

/// Why this Component must refuse to build a graph for `facts`, if it must.
/// A scheme the Component does not apply is never silently ignored.
pub fn unsupported_reason(facts: &SupportFacts) -> Option<String> {
    if facts.attention_bias {
        return Some("DeepSeek attention projections carry no bias".into());
    }
    if facts.num_attention_heads == 0
        || facts.num_key_value_heads == 0
        || !facts
            .num_attention_heads
            .is_multiple_of(facts.num_key_value_heads)
    {
        return Some(format!(
            "attention heads ({}) must be a positive multiple of key/value heads ({})",
            facts.num_attention_heads, facts.num_key_value_heads
        ));
    }
    if facts.rope_scaling == RopeScalingFacts::Banded {
        return Some("banded (llama3) RoPE scaling is not applied".into());
    }
    if facts
        .rope_scaling_factor
        .is_some_and(|factor| factor != 1.0)
    {
        return Some("a linear RoPE scaling factor is not applied".into());
    }
    if facts.has_layer_layout {
        return Some("per-layer attention geometry is not applied".into());
    }
    if facts.has_final_logit_softcapping {
        return Some("final logit softcapping is not applied".into());
    }
    if facts.has_per_layer_input {
        return Some("per-layer input embeddings are not applied".into());
    }
    if let Some(mla) = &facts.mla {
        if mla.kv_lora_rank == 0
            || mla.qk_nope_head_dim == 0
            || mla.qk_rope_head_dim == 0
            || mla.v_head_dim == 0
        {
            return Some("latent attention widths must be positive".into());
        }
        if facts.head_dimension != mla.qk_nope_head_dim + mla.qk_rope_head_dim {
            return Some(format!(
                "head dimension {} is not qk-nope {} + qk-rope {}",
                facts.head_dimension, mla.qk_nope_head_dim, mla.qk_rope_head_dim
            ));
        }
        if facts.num_key_value_heads != facts.num_attention_heads {
            return Some("latent attention has one key/value head per attention head".into());
        }
    }
    if let Some(layers) = &facts.moe_layers
        && layers.len() != facts.num_hidden_layers as usize
    {
        return Some(format!(
            "mixture-of-experts-layers has {} entries for {} layers",
            layers.len(),
            facts.num_hidden_layers
        ));
    }
    let any_moe_layer = match &facts.moe_layers {
        Some(layers) => layers.iter().any(|is_moe| *is_moe),
        None => facts.moe.is_some(),
    };
    if any_moe_layer && facts.moe.is_none() {
        return Some("a layer is a mixture of experts but none is declared".into());
    }
    if let Some(moe) = &facts.moe {
        if moe.experts == 0 || moe.experts_per_token == 0 || moe.experts_per_token > moe.experts {
            return Some(format!(
                "invalid mixture of experts: top-{} of {} experts",
                moe.experts_per_token, moe.experts
            ));
        }
        match &facts.routing {
            // The 1.5.0 routing gates a declared shared expert, DeepSeek's
            // never is: building it would not match the weights.
            None if moe.has_shared_expert => {
                return Some(
                    "no expert routing declared: the default gates the shared expert".into(),
                );
            }
            None => {}
            Some(routing) => {
                if routing.sigmoid {
                    return Some("sigmoid expert scoring is not applied".into());
                }
                if routing.has_groups {
                    return Some("group-limited expert routing is not applied".into());
                }
                if routing.selection_bias {
                    return Some("expert selection bias is not applied".into());
                }
                if routing.shared_expert_gated && moe.has_shared_expert {
                    return Some("a gated shared expert is not applied".into());
                }
                if !routing.routed_scaling_factor.is_finite()
                    || routing.routed_scaling_factor <= 0.0
                {
                    return Some("the routed scaling factor must be positive".into());
                }
            }
        }
    }
    None
}

/// Whether `layer`'s feed-forward is the declared mixture of experts.
pub fn layer_is_moe(moe_layers: Option<&[bool]>, has_moe: bool, layer: u64) -> bool {
    match moe_layers {
        Some(layers) => layers.get(layer as usize).copied().unwrap_or(false),
        None => has_moe,
    }
}

/// `yarn_get_mscale`: the softmax rescaling YaRN applies for `mscale_all_dim`.
pub fn yarn_mscale(factor: f64, mscale: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * mscale * factor.ln() + 1.0
    }
}

/// The factor applied to `QK^T`: `1/sqrt(head_dimension)`, times YaRN's
/// `mscale(factor, mscale_all_dim)^2` when the checkpoint declares one.
pub fn attention_scale(head_dimension: u64, yarn: Option<(f64, Option<f64>)>) -> f64 {
    let base = 1.0 / (head_dimension as f64).sqrt();
    match yarn {
        Some((factor, Some(mscale_all_dim))) if mscale_all_dim != 0.0 => {
            let mscale = yarn_mscale(factor, mscale_all_dim);
            base * mscale * mscale
        }
        _ => base,
    }
}

// ---------------------------------------------------------------------------
// The bound configuration, read once per build call.
// ---------------------------------------------------------------------------

struct Plan {
    config: ArchitectureConfig,
    moe: Option<MoeConfig>,
    moe_layers: Option<Vec<bool>>,
    mla: Option<MlaConfig>,
    routing: Option<ExpertRoutingConfig>,
    yarn: Option<YarnRopeScaling>,
}

impl Plan {
    /// Reads the `model-config` features this Component supports (the
    /// Runtime refuses a graph for a checkpoint whose declared feature was
    /// never read), and refuses what this Component cannot build.
    fn read() -> Self {
        match model_config::hidden_activation().as_deref() {
            Some("silu") => {}
            Some(other) => panic!("the DeepSeek Component does not support activation {other:?}"),
            None => panic!("the DeepSeek Component requires a known checkpoint activation"),
        }
        let config = model_config::model_config();
        let options = model_config::options();
        let moe = model_config::mixture_of_experts();
        let moe_layers = model_config::mixture_of_experts_layers();
        let mla = model_config::multi_head_latent_attention();
        let routing = model_config::expert_routing();
        let (scaling, yarn) = match model_config::rope_scaling() {
            None => (RopeScalingFacts::None, None),
            Some(RopeFrequencyScaling::Banded(_)) => (RopeScalingFacts::Banded, None),
            Some(RopeFrequencyScaling::Yarn(yarn)) => (RopeScalingFacts::Yarn, Some(yarn)),
        };
        let facts = SupportFacts {
            attention_bias: config.attention_bias,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dimension: config.head_dimension,
            rope_scaling_factor: config.rope_scaling_factor,
            rope_scaling: scaling,
            has_layer_layout: model_config::layer_layout().is_some(),
            has_final_logit_softcapping: options.final_logit_softcapping.is_some(),
            has_per_layer_input: options.per_layer_input_size.is_some(),
            mla: mla.as_ref().map(|mla| MlaFacts {
                kv_lora_rank: mla.kv_lora_rank,
                qk_nope_head_dim: mla.qk_nope_head_dim,
                qk_rope_head_dim: mla.qk_rope_head_dim,
                v_head_dim: mla.v_head_dim,
            }),
            moe: moe.as_ref().map(|moe| MoeFacts {
                experts: moe.experts,
                experts_per_token: moe.experts_per_token,
                has_shared_expert: moe.shared_expert_intermediate_size.is_some(),
            }),
            moe_layers: moe_layers.clone(),
            routing: routing.as_ref().map(|routing| RoutingFacts {
                sigmoid: matches!(routing.scoring, ExpertScoring::Sigmoid),
                has_groups: routing.groups.is_some(),
                routed_scaling_factor: routing.routed_scaling_factor,
                selection_bias: routing.selection_bias,
                shared_expert_gated: routing.shared_expert_gated,
            }),
        };
        if let Some(reason) = unsupported_reason(&facts) {
            panic!("the DeepSeek Component cannot build this configuration: {reason}");
        }
        Self {
            config,
            moe,
            moe_layers,
            mla,
            routing,
            yarn,
        }
    }

    fn layer_moe(&self, layer: u64) -> Option<&MoeConfig> {
        let is_moe = layer_is_moe(self.moe_layers.as_deref(), self.moe.is_some(), layer);
        self.moe.as_ref().filter(|_| is_moe)
    }

    /// The `rope` node attributes for `head_count` blocks of `dimension`,
    /// YaRN included when declared: every node states exactly the
    /// checkpoint's frequency scaling.
    fn rope_attrs(&self, dimension: u64, head_count: u64, pairing: &str) -> Vec<Attribute> {
        let mut attributes = vec![
            attr("base", AttributeValue::FloatValue(self.config.rope_theta)),
            attr("dimension", AttributeValue::Integer(dimension as i64)),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
            ),
            attr(
                "rotation_pairing",
                AttributeValue::Text(pairing.to_string()),
            ),
            attr("position_offset", AttributeValue::Integer(0)),
            attr("head_count", AttributeValue::Integer(head_count as i64)),
        ];
        if let Some(yarn) = &self.yarn {
            attributes.extend([
                attr(
                    "frequency_scaling",
                    AttributeValue::Text("yarn".to_string()),
                ),
                attr(
                    "frequency_scaling_factor",
                    AttributeValue::FloatValue(yarn.factor),
                ),
                attr(
                    "original_context_length",
                    AttributeValue::Integer(yarn.original_max_position_embeddings as i64),
                ),
                attr("beta_fast", AttributeValue::FloatValue(yarn.beta_fast)),
                attr("beta_slow", AttributeValue::FloatValue(yarn.beta_slow)),
                attr(
                    "attention_factor",
                    AttributeValue::FloatValue(yarn.attention_factor),
                ),
                attr("truncate", AttributeValue::Boolean(yarn.truncate)),
            ]);
        }
        attributes
    }

    fn softmax_scale(&self, head_dimension: u64) -> f64 {
        attention_scale(
            head_dimension,
            self.yarn.as_ref().map(|y| (y.factor, y.mscale_all_dim)),
        )
    }
}

// ---------------------------------------------------------------------------
// Graph construction.
// ---------------------------------------------------------------------------

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.to_string(),
        value,
    }
}

/// How a node may be split across Devices: `column` and `row` matmuls,
/// `local` elementwise nodes between them, `experts` whole routed experts.
fn tensor_parallel(kind: &str) -> Attribute {
    attr("tensor_parallel", AttributeValue::Text(kind.to_string()))
}

/// The canonical Model Artifact tensor name `weight_edge` is called with
/// for a per-layer weight (the Runtime strips a generic prefix; it carries
/// no DeepSeek-specific mapping table). Latent attention's split projections
/// are the names the loaders produce (`latent_attention_projection_shapes`).
pub fn weight_tensor_name(layer: u64, suffix: &str) -> String {
    match suffix {
        "input_norm" | "post_attn_norm" => format!("layers.{layer}.{suffix}"),
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_a_proj" | "q_a_norm" | "q_nope_proj"
        | "q_rope_proj" | "kv_a_proj" | "kv_a_norm" | "k_rope_proj" | "k_nope_proj"
        | "k_absorb" | "v_absorb" => {
            format!("layers.{layer}.self_attn.{suffix}")
        }
        "gate_proj" | "up_proj" | "down_proj" => format!("layers.{layer}.mlp.{suffix}"),
        "router_proj" | "experts_gate_proj" | "experts_up_proj" | "experts_down_proj"
        | "shared_gate_proj" | "shared_up_proj" | "shared_down_proj" => {
            format!("layers.{layer}.moe.{suffix}")
        }
        other => unreachable!("weight_tensor_name: unknown weight suffix '{other}'"),
    }
}

/// One matmul `input x weight(layer, suffix)` producing `[sequence, cols]`.
#[allow(clippy::too_many_arguments)]
fn matmul(
    prefix: &str,
    layer: u64,
    name: &str,
    suffix: &str,
    input: &str,
    rows: u64,
    cols: u64,
    parallel: Option<&str>,
    kv: Option<&KvInvolvement>,
) -> String {
    let attributes: Vec<Attribute> = parallel.map(tensor_parallel).into_iter().collect();
    add_node(
        &format!("{prefix}.{name}"),
        "matmul",
        "LinearAlgebra",
        &attributes,
        &[
            input.to_string(),
            weight_edge(&weight_tensor_name(layer, suffix)),
        ],
        &shape(vec![rows, cols]),
        kv,
    )
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm(
    plan: &Plan,
    prefix: &str,
    layer: u64,
    name: &str,
    suffix: &str,
    input: &str,
    rows: u64,
    cols: u64,
) -> String {
    add_node(
        &format!("{prefix}.{name}"),
        "rmsnorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(plan.config.rms_norm_eps),
        )],
        &[
            input.to_string(),
            weight_edge(&weight_tensor_name(layer, suffix)),
        ],
        &shape(vec![rows, cols]),
        None,
    )
}

fn kv_involvement(layer: u64, role: KvCacheRole, append: bool) -> KvInvolvement {
    KvInvolvement {
        resource_id: kv_resource(layer, role),
        role,
        append,
    }
}

/// Attention over the normalized hidden state `normed`: returns the
/// attention output `[sequence, heads * value width]` and that width.
fn build_attention(
    plan: &Plan,
    layer: u64,
    prefix: &str,
    normed: &str,
    sequence_length: u64,
    kv_append: bool,
) -> (String, u64) {
    match &plan.mla {
        None => build_standard_attention(plan, layer, prefix, normed, sequence_length, kv_append),
        Some(mla) => {
            build_latent_attention(plan, mla, layer, prefix, normed, sequence_length, kv_append)
        }
    }
}

/// Multi/grouped-query attention with half-split RoPE.
fn build_standard_attention(
    plan: &Plan,
    layer: u64,
    prefix: &str,
    normed: &str,
    sequence_length: u64,
    kv_append: bool,
) -> (String, u64) {
    let config = &plan.config;
    let heads = u64::from(config.num_attention_heads);
    let kv_heads = u64::from(config.num_key_value_heads);
    let head_dimension = config.head_dimension;
    let q_dim = heads * head_dimension;
    let kv_dim = kv_heads * head_dimension;

    let q = matmul(
        prefix,
        layer,
        "q_proj",
        "q_proj",
        normed,
        sequence_length,
        q_dim,
        None,
        None,
    );
    let k = matmul(
        prefix,
        layer,
        "k_proj",
        "k_proj",
        normed,
        sequence_length,
        kv_dim,
        None,
        None,
    );
    let v = matmul(
        prefix,
        layer,
        "v_proj",
        "v_proj",
        normed,
        sequence_length,
        kv_dim,
        None,
        Some(&kv_involvement(layer, KvCacheRole::V, kv_append)),
    );
    // Hugging Face DeepSeek rotates (i, i + dimension / 2).
    let q_rope = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &plan.rope_attrs(head_dimension, heads, "half_split"),
        &[q],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let k_rope = add_node(
        &format!("{prefix}.rope_k"),
        "rope",
        "PositionEncoding",
        &plan.rope_attrs(head_dimension, kv_heads, "half_split"),
        &[k],
        &shape(vec![sequence_length, kv_dim]),
        Some(&kv_involvement(layer, KvCacheRole::K, kv_append)),
    );
    let mut attributes = attention_attrs(heads, kv_heads, head_dimension);
    if plan.yarn.is_some() {
        attributes.push(attr(
            "scale",
            AttributeValue::FloatValue(plan.softmax_scale(head_dimension)),
        ));
    }
    let out = add_node(
        &format!("{prefix}.attention"),
        "attention",
        "Attention",
        &attributes,
        &[q_rope, k_rope, v],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    (out, q_dim)
}

fn attention_attrs(heads: u64, kv_heads: u64, head_dimension: u64) -> Vec<Attribute> {
    vec![
        attr("causal", AttributeValue::Boolean(true)),
        attr("head_count", AttributeValue::Integer(heads as i64)),
        attr("kv_head_count", AttributeValue::Integer(kv_heads as i64)),
        attr(
            "head_dimension",
            AttributeValue::Integer(head_dimension as i64),
        ),
        attr(
            "attention_mask_kind",
            AttributeValue::Text("causal".to_string()),
        ),
    ]
}

/// Multi-head Latent Attention (`cache-latent-attention-history`). The
/// loaders have split every fused projection by rows. The KV history is
/// the latent row `[c_kv | k_pe]` (`kv_lora_rank + qk_rope_head_dim`
/// values, one key head) and nothing else:
/// - a prefill writes the latent rows to the history and attends over it
///   with each head's expanded key `[c_kv W_UK | k_pe]` and value
///   `c_kv W_UV` (`latent_kv_width`: the Provider expands the rows it
///   reads; fewer operations than attending in the latent width), so its
///   passes over a long prompt see the earlier ones (`chunk-latent-prefills`);
/// - a decode step absorbs each head's position-independent query into the
///   latent space (`q_nope W_UK^T`, `batched-matmul` by `k_absorb`),
///   attends over the latent history as one key head whose leading
///   `kv_lora_rank` values are also the values (`value_head_stride`), and
///   up-projects each head's latent output (`batched-matmul` by
///   `v_absorb`). Its scores are the expanded form's, with the same scale.
fn build_latent_attention(
    plan: &Plan,
    mla: &MlaConfig,
    layer: u64,
    prefix: &str,
    normed: &str,
    sequence_length: u64,
    kv_append: bool,
) -> (String, u64) {
    let t = sequence_length;
    let heads = u64::from(plan.config.num_attention_heads);
    let (nope, rope, value) = (mla.qk_nope_head_dim, mla.qk_rope_head_dim, mla.v_head_dim);
    let qk = nope + rope;
    let latent = u64::from(mla.kv_lora_rank);

    // Queries: from the hidden state, or through a low-rank bottleneck.
    let query_input = match mla.q_lora_rank {
        Some(rank) => {
            let rank = u64::from(rank);
            let a = matmul(
                prefix, layer, "q_a_proj", "q_a_proj", normed, t, rank, None, None,
            );
            rmsnorm(plan, prefix, layer, "q_a_norm", "q_a_norm", &a, t, rank)
        }
        None => normed.to_string(),
    };
    let q_nope = matmul(
        prefix,
        layer,
        "q_nope_proj",
        "q_nope_proj",
        &query_input,
        t,
        heads * nope,
        None,
        None,
    );
    let q_pe = matmul(
        prefix,
        layer,
        "q_rope_proj",
        "q_rope_proj",
        &query_input,
        t,
        heads * rope,
        None,
        None,
    );
    // DeepSeek-V2/V3 rotate adjacent pairs (`view_as_complex`).
    let q_pe = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &plan.rope_attrs(rope, heads, "interleaved"),
        &[q_pe],
        &shape(vec![t, heads * rope]),
        None,
    );

    // The normalized latent and the one shared rotated key.
    let c_kv = matmul(
        prefix,
        layer,
        "kv_a_proj",
        "kv_a_proj",
        normed,
        t,
        latent,
        None,
        None,
    );
    let c_kv = rmsnorm(
        plan,
        prefix,
        layer,
        "kv_a_norm",
        "kv_a_norm",
        &c_kv,
        t,
        latent,
    );
    let k_pe = matmul(
        prefix,
        layer,
        "k_rope_proj",
        "k_rope_proj",
        normed,
        t,
        rope,
        None,
        None,
    );
    let k_pe = add_node(
        &format!("{prefix}.rope_k"),
        "rope",
        "PositionEncoding",
        &plan.rope_attrs(rope, 1, "interleaved"),
        &[k_pe],
        &shape(vec![t, rope]),
        None,
    );
    // The history: `[c_kv | k_pe]` per token, one head.
    let history = add_node(
        &format!("{prefix}.latent"),
        "concat",
        "Tensor",
        &head_concat_attrs(1, 1),
        &[c_kv.clone(), k_pe.clone()],
        &shape(vec![t, latent + rope]),
        Some(&kv_involvement(layer, KvCacheRole::K, kv_append)),
    );
    let scale = attr("scale", AttributeValue::FloatValue(plan.softmax_scale(qk)));

    if !kv_append {
        // Prefill: attends over the latent history its `latent` node
        // writes (its own rows after any earlier pass's), each head's key
        // `[c_kv W_UK | k_pe]` and value `c_kv W_UV` expanded by the
        // Provider (`latent_kv_width`, `chunk-latent-prefills`), so a long
        // prompt can run in passes that see the earlier ones.
        let q = head_concat(prefix, "q", &q_nope, &q_pe, heads, heads, t, qk);
        let mut attributes = attention_attrs(heads, heads, qk);
        attributes.push(attr(
            "value_head_dimension",
            AttributeValue::Integer(value as i64),
        ));
        attributes.push(attr(
            "latent_kv_width",
            AttributeValue::Integer(latent as i64),
        ));
        attributes.push(scale);
        let out = add_node(
            &format!("{prefix}.attention"),
            "attention",
            "Attention",
            &attributes,
            &[
                q,
                history.clone(),
                history,
                weight_edge(&weight_tensor_name(layer, "k_nope_proj")),
                weight_edge(&weight_tensor_name(layer, "v_proj")),
            ],
            &shape(vec![t, heads * value]),
            None,
        );
        return (out, heads * value);
    }

    // Decode: absorbed queries over the latent history.
    let q_latent = add_node(
        &format!("{prefix}.q_absorbed"),
        "batched-matmul",
        "LinearAlgebra",
        &[attr("group_count", AttributeValue::Integer(heads as i64))],
        &[q_nope, weight_edge(&weight_tensor_name(layer, "k_absorb"))],
        &shape(vec![t, heads * latent]),
        None,
    );
    let q = head_concat(
        prefix,
        "q",
        &q_latent,
        &q_pe,
        heads,
        heads,
        t,
        latent + rope,
    );
    let mut attributes = attention_attrs(heads, 1, latent + rope);
    attributes.push(attr(
        "value_head_dimension",
        AttributeValue::Integer(latent as i64),
    ));
    attributes.push(attr(
        "value_head_stride",
        AttributeValue::Integer((latent + rope) as i64),
    ));
    attributes.push(scale);
    let latent_out = add_node(
        &format!("{prefix}.attention"),
        "attention",
        "Attention",
        &attributes,
        &[q, history.clone(), history],
        &shape(vec![t, heads * latent]),
        None,
    );
    let out = add_node(
        &format!("{prefix}.v_absorbed"),
        "batched-matmul",
        "LinearAlgebra",
        &[
            attr("group_count", AttributeValue::Integer(heads as i64)),
            attr("transpose_b", AttributeValue::Boolean(true)),
        ],
        &[
            latent_out,
            weight_edge(&weight_tensor_name(layer, "v_absorb")),
        ],
        &shape(vec![t, heads * value]),
        None,
    );
    (out, heads * value)
}

fn head_concat_attrs(head_count: u64, second_head_count: u64) -> Vec<Attribute> {
    vec![
        attr("axis", AttributeValue::Text("heads".to_string())),
        attr("head_count", AttributeValue::Integer(head_count as i64)),
        attr(
            "second_head_count",
            AttributeValue::Integer(second_head_count as i64),
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn head_concat(
    prefix: &str,
    name: &str,
    first: &str,
    second: &str,
    head_count: u64,
    second_head_count: u64,
    sequence_length: u64,
    head_width: u64,
) -> String {
    add_node(
        &format!("{prefix}.{name}"),
        "concat",
        "Tensor",
        &head_concat_attrs(head_count, second_head_count),
        &[first.to_string(), second.to_string()],
        &shape(vec![sequence_length, head_count * head_width]),
        None,
    )
}

/// Builds one pre-norm decoder layer: RMSNorm, attention, output projection,
/// residual, RMSNorm, MLP, residual.
fn build_layer(
    plan: &Plan,
    layer: u64,
    residual_in: &str,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    let prefix = format!("layer{layer}");
    let hidden_size = plan.config.hidden_size;

    let normed = rmsnorm(
        plan,
        &prefix,
        layer,
        "input_norm",
        "input_norm",
        residual_in,
        sequence_length,
        hidden_size,
    );
    let (attn_out, attn_width) =
        build_attention(plan, layer, &prefix, &normed, sequence_length, kv_append);
    let attn_proj = matmul(
        &prefix,
        layer,
        "o_proj",
        "o_proj",
        &attn_out,
        sequence_length,
        hidden_size,
        None,
        None,
    );
    debug_assert!(attn_width > 0);
    let post_attn = add_node(
        &format!("{prefix}.residual1"),
        "residual-add",
        "Tensor",
        &[],
        &[residual_in.to_string(), attn_proj],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let mlp_normed = rmsnorm(
        plan,
        &prefix,
        layer,
        "post_attn_norm",
        "post_attn_norm",
        &post_attn,
        sequence_length,
        hidden_size,
    );
    let mlp_out = match plan.layer_moe(layer) {
        None => gated_mlp(
            &prefix,
            layer,
            "",
            &mlp_normed,
            sequence_length,
            hidden_size,
            plan.config.intermediate_size,
        ),
        Some(moe) => moe_mlp(
            plan,
            &prefix,
            layer,
            moe,
            &mlp_normed,
            sequence_length,
            hidden_size,
        ),
    };
    add_node(
        &format!("{prefix}.residual2"),
        "residual-add",
        "Tensor",
        &[],
        &[post_attn, mlp_out],
        &shape(vec![sequence_length, hidden_size]),
        None,
    )
}

/// `down(silu(gate(h)) * up(h))`, tensor-parallel: `gate`/`up` by columns,
/// `silu`/`mul` local, `down` by rows. `shared` is `"shared_"` for the
/// shared expert's weights and node names, `""` for a dense MLP.
fn gated_mlp(
    prefix: &str,
    layer: u64,
    shared: &str,
    input: &str,
    sequence_length: u64,
    hidden_size: u64,
    intermediate_size: u64,
) -> String {
    let (gate_w, up_w, down_w) = (
        format!("{shared}gate_proj"),
        format!("{shared}up_proj"),
        format!("{shared}down_proj"),
    );
    let gate = matmul(
        prefix,
        layer,
        &gate_w,
        &gate_w,
        input,
        sequence_length,
        intermediate_size,
        Some("column"),
        None,
    );
    let up = matmul(
        prefix,
        layer,
        &up_w,
        &up_w,
        input,
        sequence_length,
        intermediate_size,
        Some("column"),
        None,
    );
    let local = |name: &str, operator: &str, family: &str, inputs: &[String]| {
        add_node(
            &format!("{prefix}.{shared}{name}"),
            operator,
            family,
            &[tensor_parallel("local")],
            inputs,
            &shape(vec![sequence_length, intermediate_size]),
            None,
        )
    };
    let activated = local("silu", "silu", "Activation", &[gate]);
    let hidden = local("mul", "mul", "Tensor", &[activated, up]);
    matmul(
        prefix,
        layer,
        &down_w,
        &down_w,
        &hidden,
        sequence_length,
        hidden_size,
        Some("row"),
        None,
    )
}

/// DeepSeekMoE: `moe-experts(h, router(h), experts)`, scaled by the routed
/// scaling factor, plus -- when the checkpoint has shared experts -- their
/// plain (ungated) gated MLP added to the routed output.
fn moe_mlp(
    plan: &Plan,
    prefix: &str,
    layer: u64,
    moe: &MoeConfig,
    input: &str,
    sequence_length: u64,
    hidden_size: u64,
) -> String {
    // The router is replicated: every Device routes over every expert.
    let logits = matmul(
        prefix,
        layer,
        "router",
        "router_proj",
        input,
        sequence_length,
        u64::from(moe.experts),
        None,
        None,
    );
    let mut routed = add_node(
        &format!("{prefix}.experts"),
        "moe-experts",
        "LinearAlgebra",
        &[
            attr("experts", AttributeValue::Integer(i64::from(moe.experts))),
            attr(
                "top_k",
                AttributeValue::Integer(i64::from(moe.experts_per_token)),
            ),
            attr(
                "normalize_top_k",
                AttributeValue::Boolean(moe.normalize_top_k),
            ),
            tensor_parallel("experts"),
        ],
        &[
            input.to_string(),
            logits,
            weight_edge(&weight_tensor_name(layer, "experts_gate_proj")),
            weight_edge(&weight_tensor_name(layer, "experts_up_proj")),
            weight_edge(&weight_tensor_name(layer, "experts_down_proj")),
        ],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );
    // Scaling every routing weight by `s` scales the routed output by `s`.
    if let Some(routing) = &plan.routing
        && routing.routed_scaling_factor != 1.0
    {
        routed = add_node(
            &format!("{prefix}.routed_scale"),
            "activation",
            "Activation",
            &[
                attr("kind", AttributeValue::Text("scale".to_string())),
                attr(
                    "factor",
                    AttributeValue::FloatValue(routing.routed_scaling_factor),
                ),
            ],
            &[routed],
            &shape(vec![sequence_length, hidden_size]),
            None,
        );
    }
    let Some(shared_size) = moe.shared_expert_intermediate_size else {
        return routed;
    };
    let shared = gated_mlp(
        prefix,
        layer,
        "shared_",
        input,
        sequence_length,
        hidden_size,
        shared_size,
    );
    add_node(
        &format!("{prefix}.moe_out"),
        "residual-add",
        "Tensor",
        &[],
        &[routed, shared],
        &shape(vec![sequence_length, hidden_size]),
        None,
    )
}

/// Builds decoder layers `[start_layer, end_layer)`. `build_graph` is
/// exactly `build_graph_segment(.., 0, num_hidden_layers)`.
fn build_graph_segment(
    phase: GraphPhase,
    sequence_length: u64,
    position_offset: u64,
    kv_append: bool,
    start_layer: u32,
    end_layer: u32,
) -> String {
    let plan = Plan::read();
    let config = &plan.config;
    begin_graph(phase, sequence_length, position_offset);

    let mut hidden = if start_layer == 0 {
        let token_ids = declare_input("token_ids", &shape(vec![sequence_length]));
        add_node(
            "embedding",
            "embedding",
            "Tensor",
            &[],
            &[token_ids, weight_edge("token_embedding")],
            &shape(vec![sequence_length, config.hidden_size]),
            None,
        )
    } else {
        declare_input(
            "hidden_states_in",
            &shape(vec![sequence_length, config.hidden_size]),
        )
    };

    for layer in u64::from(start_layer)..u64::from(end_layer) {
        hidden = build_layer(&plan, layer, &hidden, sequence_length, kv_append);
    }

    if end_layer != config.num_hidden_layers {
        return finish_graph(&hidden);
    }
    let hidden_final = add_node(
        "final_norm",
        "rmsnorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(config.rms_norm_eps),
        )],
        &[hidden, weight_edge("final_norm")],
        &shape(vec![sequence_length, config.hidden_size]),
        None,
    );
    let lm_head_weight = weight_edge("lm_head");
    if config.tie_word_embeddings {
        // The alias target must already exist in this graph; a segment that
        // does not start at layer 0 has not registered it yet.
        weight_edge("token_embedding");
        alias_weight_edge(&lm_head_weight, "token_embedding");
    }
    let logits = add_node(
        "lm_head",
        "matmul",
        "LinearAlgebra",
        &[],
        &[hidden_final, lm_head_weight],
        &shape(vec![sequence_length, config.vocabulary_size]),
        None,
    );
    finish_graph(&logits)
}

fn build_graph(
    phase: GraphPhase,
    sequence_length: u64,
    position_offset: u64,
    kv_append: bool,
) -> String {
    let layers = model_config::model_config().num_hidden_layers;
    build_graph_segment(
        phase,
        sequence_length,
        position_offset,
        kv_append,
        0,
        layers,
    )
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        build_graph(GraphPhase::Prefill, sequence_length, 0, false)
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        build_graph(GraphPhase::Decode, 1, cached_token_count, true)
    }

    fn build_prefill_graph_segment(
        sequence_length: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build_graph_segment(
            GraphPhase::Prefill,
            sequence_length,
            0,
            false,
            start_layer,
            end_layer,
        )
    }

    fn build_decode_graph_segment(
        cached_token_count: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build_graph_segment(
            GraphPhase::Decode,
            1,
            cached_token_count,
            true,
            start_layer,
            end_layer,
        )
    }
}

export!(Component);

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-V2-Lite: latent attention without a query bottleneck, a dense
    /// layer 0 then MoE layers with two shared experts, softmax routing.
    fn v2_lite() -> SupportFacts {
        SupportFacts {
            attention_bias: false,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dimension: 192,
            rope_scaling_factor: None,
            rope_scaling: RopeScalingFacts::Yarn,
            has_layer_layout: false,
            has_final_logit_softcapping: false,
            has_per_layer_input: false,
            mla: Some(MlaFacts {
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
            }),
            moe: Some(MoeFacts {
                experts: 64,
                experts_per_token: 6,
                has_shared_expert: true,
            }),
            moe_layers: Some((0..27).map(|layer| layer > 0).collect()),
            routing: Some(RoutingFacts {
                sigmoid: false,
                has_groups: false,
                routed_scaling_factor: 1.0,
                selection_bias: false,
                shared_expert_gated: false,
            }),
        }
    }

    /// DeepSeek-MoE-16B: ordinary attention, a dense layer 0, then MoE.
    fn moe_16b() -> SupportFacts {
        SupportFacts {
            num_hidden_layers: 28,
            head_dimension: 128,
            rope_scaling: RopeScalingFacts::None,
            mla: None,
            moe_layers: Some((0..28).map(|layer| layer > 0).collect()),
            ..v2_lite()
        }
    }

    fn refused(facts: SupportFacts) {
        assert!(unsupported_reason(&facts).is_some(), "{facts:?}");
    }

    #[test]
    fn names_its_architecture_family() {
        assert_eq!(ARCHITECTURE_FAMILY, "deepseek");
    }

    #[test]
    fn accepts_the_deepseek_shapes_it_builds() {
        assert_eq!(unsupported_reason(&v2_lite()), None);
        assert_eq!(unsupported_reason(&moe_16b()), None);
        let dense = SupportFacts {
            moe: None,
            moe_layers: None,
            routing: None,
            mla: None,
            head_dimension: 128,
            num_key_value_heads: 4,
            rope_scaling: RopeScalingFacts::None,
            ..v2_lite()
        };
        assert_eq!(unsupported_reason(&dense), None);
        let scaled = SupportFacts {
            routing: Some(RoutingFacts {
                routed_scaling_factor: 16.0,
                ..v2_lite().routing.unwrap()
            }),
            ..v2_lite()
        };
        assert_eq!(unsupported_reason(&scaled), None);
        let unit_factor = SupportFacts {
            rope_scaling_factor: Some(1.0),
            ..v2_lite()
        };
        assert_eq!(unsupported_reason(&unit_factor), None);
    }

    #[test]
    fn refuses_what_it_does_not_apply() {
        refused(SupportFacts {
            attention_bias: true,
            ..v2_lite()
        });
        refused(SupportFacts {
            rope_scaling: RopeScalingFacts::Banded,
            ..v2_lite()
        });
        refused(SupportFacts {
            rope_scaling_factor: Some(4.0),
            ..v2_lite()
        });
        refused(SupportFacts {
            has_layer_layout: true,
            ..v2_lite()
        });
        refused(SupportFacts {
            has_final_logit_softcapping: true,
            ..v2_lite()
        });
        refused(SupportFacts {
            has_per_layer_input: true,
            ..v2_lite()
        });
        refused(SupportFacts {
            num_key_value_heads: 5,
            ..moe_16b()
        });
    }

    #[test]
    fn refuses_inconsistent_latent_attention() {
        let mla = v2_lite().mla.unwrap();
        refused(SupportFacts {
            head_dimension: 128,
            ..v2_lite()
        });
        refused(SupportFacts {
            num_key_value_heads: 8,
            ..v2_lite()
        });
        refused(SupportFacts {
            mla: Some(MlaFacts {
                v_head_dim: 0,
                ..mla
            }),
            ..v2_lite()
        });
    }

    #[test]
    fn refuses_routing_it_does_not_apply() {
        let routing = v2_lite().routing.unwrap();
        for changed in [
            RoutingFacts {
                sigmoid: true,
                ..routing.clone()
            },
            RoutingFacts {
                has_groups: true,
                ..routing.clone()
            },
            RoutingFacts {
                selection_bias: true,
                ..routing.clone()
            },
            RoutingFacts {
                shared_expert_gated: true,
                ..routing.clone()
            },
            RoutingFacts {
                routed_scaling_factor: 0.0,
                ..routing.clone()
            },
        ] {
            refused(SupportFacts {
                routing: Some(changed),
                ..v2_lite()
            });
        }
        // The default routing gates a declared shared expert.
        refused(SupportFacts {
            routing: None,
            ..v2_lite()
        });
        // Without a shared expert, the default routing is the one built.
        let no_shared = SupportFacts {
            routing: None,
            moe: Some(MoeFacts {
                has_shared_expert: false,
                ..v2_lite().moe.unwrap()
            }),
            ..v2_lite()
        };
        assert_eq!(unsupported_reason(&no_shared), None);
    }

    #[test]
    fn refuses_inconsistent_expert_layers() {
        refused(SupportFacts {
            moe_layers: Some(vec![true; 3]),
            ..v2_lite()
        });
        refused(SupportFacts {
            moe: None,
            routing: None,
            ..v2_lite()
        });
        refused(SupportFacts {
            moe: Some(MoeFacts {
                experts_per_token: 65,
                ..v2_lite().moe.unwrap()
            }),
            ..v2_lite()
        });
    }

    #[test]
    fn picks_each_layers_feed_forward() {
        let layers = [false, true, true];
        assert!(!layer_is_moe(Some(&layers), true, 0));
        assert!(layer_is_moe(Some(&layers), true, 2));
        assert!(layer_is_moe(None, true, 5));
        assert!(!layer_is_moe(None, false, 5));
    }

    #[test]
    fn rescales_the_softmax_as_yarn_does() {
        // Without YaRN, or without `mscale_all_dim`: 1/sqrt(head_dimension).
        let base = 1.0 / 192f64.sqrt();
        assert_eq!(attention_scale(192, None), base);
        assert_eq!(attention_scale(192, Some((40.0, None))), base);
        assert_eq!(attention_scale(192, Some((40.0, Some(0.0)))), base);
        // DeepSeek-V2-Lite: factor 40, mscale_all_dim 0.707.
        let mscale = 0.1 * 0.707 * 40f64.ln() + 1.0;
        let scale = attention_scale(192, Some((40.0, Some(0.707))));
        assert!((scale - base * mscale * mscale).abs() < 1e-12);
        assert!(scale > base);
        // A factor of at most one never rescales.
        assert_eq!(yarn_mscale(1.0, 0.707), 1.0);
        assert_eq!(attention_scale(192, Some((1.0, Some(0.707)))), base);
    }

    #[test]
    fn weight_names_follow_the_loaders_layout() {
        assert_eq!(weight_tensor_name(3, "q_proj"), "layers.3.self_attn.q_proj");
        for latent in [
            "q_a_proj",
            "q_a_norm",
            "q_nope_proj",
            "q_rope_proj",
            "kv_a_proj",
            "kv_a_norm",
            "k_rope_proj",
            "k_nope_proj",
        ] {
            assert_eq!(
                weight_tensor_name(1, latent),
                format!("layers.1.self_attn.{latent}")
            );
        }
        assert_eq!(weight_tensor_name(3, "down_proj"), "layers.3.mlp.down_proj");
        assert_eq!(
            weight_tensor_name(3, "shared_up_proj"),
            "layers.3.moe.shared_up_proj"
        );
        assert_eq!(
            weight_tensor_name(0, "post_attn_norm"),
            "layers.0.post_attn_norm"
        );
    }
}
