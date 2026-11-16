//! DeepSeek Model Component for the Magnetar local AI Runtime
//! (`magnetar:model-component-graph@1.6.0`).
//!
//! Builds prefill and decode Execution Graphs for the DeepSeek decoder
//! family (`model_type = "deepseek"`, DeepSeekMoE): pre-norm decoder layers
//! with multi-head (or grouped-query) attention, half-split RoPE and no
//! projection bias, whose feed-forward is either a dense gated MLP or
//! DeepSeekMoE -- fine-grained routed experts plus an always-active shared
//! expert that, unlike Qwen's, is **not** gated by a sigmoid.
//!
//! Every dimension is queried from `model-config` on each build call, never
//! compiled in. What this Component cannot express through the contract it
//! refuses to build rather than approximating (see [`unsupported_reason`]):
//! most importantly Multi-head Latent Attention (DeepSeek-V2/V3), which the
//! contract does not model. DeepSeek-R1 *distilled* checkpoints are Qwen2.5
//! and Llama models and run through those Components, not this one.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape, add_node,
    alias_weight_edge, begin_graph, declare_input, finish_graph, kv_resource, weight_edge,
};
use magnetar::model_component_graph::model_config::{
    self, ArchitectureConfig, ArchitectureOptions, MoeConfig,
};

/// The architecture family this Component serves.
pub const ARCHITECTURE_FAMILY: &str = "deepseek";

const ROPE_POSITION_MODE: &str = "sequential";
// Hugging Face DeepSeek checkpoints rotate (i, i + dimension / 2).
const ROPE_ROTATION_PAIRING: &str = "half_split";

/// The facts about a bound configuration that decide whether this Component
/// can build a faithful graph. Plain data, so the decision is unit-testable
/// without a Runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportFacts {
    pub attention_bias: bool,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub has_rope_scaling: bool,
    pub has_layer_layout: bool,
    pub has_final_logit_softcapping: bool,
    pub has_per_layer_input: bool,
    pub moe: Option<MoeFacts>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeFacts {
    pub experts: u32,
    pub experts_per_token: u32,
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
    if facts.has_rope_scaling {
        return Some("per-frequency RoPE scaling (e.g. YaRN) is not applied".into());
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
    if let Some(moe) = &facts.moe
        && (moe.experts == 0 || moe.experts_per_token == 0 || moe.experts_per_token > moe.experts)
    {
        return Some(format!(
            "invalid mixture of experts: top-{} of {} experts",
            moe.experts_per_token, moe.experts
        ));
    }
    None
}

fn facts_of(
    config: &ArchitectureConfig,
    options: &ArchitectureOptions,
    moe: Option<&MoeConfig>,
) -> SupportFacts {
    SupportFacts {
        attention_bias: config.attention_bias,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        has_rope_scaling: model_config::rope_scaling().is_some(),
        has_layer_layout: model_config::layer_layout().is_some(),
        has_final_logit_softcapping: options.final_logit_softcapping.is_some(),
        has_per_layer_input: options.per_layer_input_size.is_some(),
        moe: moe.map(|moe| MoeFacts {
            experts: moe.experts,
            experts_per_token: moe.experts_per_token,
        }),
    }
}

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
/// no DeepSeek-specific mapping table).
pub fn weight_tensor_name(layer: u64, suffix: &str) -> String {
    match suffix {
        "input_norm" | "post_attn_norm" => format!("layers.{layer}.{suffix}"),
        "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("layers.{layer}.self_attn.{suffix}"),
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

/// Builds one pre-norm decoder layer: RMSNorm, Q/K/V projections, RoPE,
/// attention, output projection, residual, RMSNorm, MLP, residual.
fn build_layer(
    config: &ArchitectureConfig,
    moe: Option<&MoeConfig>,
    layer: u64,
    residual_in: &str,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    let prefix = format!("layer{layer}");
    let hidden_size = config.hidden_size;
    let heads = u64::from(config.num_attention_heads);
    let kv_heads = u64::from(config.num_key_value_heads);
    let head_dimension = config.head_dimension;
    let q_dim = heads * head_dimension;
    let kv_dim = kv_heads * head_dimension;
    let epsilon = || attr("epsilon", AttributeValue::FloatValue(config.rms_norm_eps));

    let normed = add_node(
        &format!("{prefix}.input_norm"),
        "rmsnorm",
        "Normalization",
        &[epsilon()],
        &[
            residual_in.to_string(),
            weight_edge(&weight_tensor_name(layer, "input_norm")),
        ],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let q = matmul(
        &prefix,
        layer,
        "q_proj",
        "q_proj",
        &normed,
        sequence_length,
        q_dim,
        None,
        None,
    );
    let k = matmul(
        &prefix,
        layer,
        "k_proj",
        "k_proj",
        &normed,
        sequence_length,
        kv_dim,
        None,
        None,
    );
    let v = matmul(
        &prefix,
        layer,
        "v_proj",
        "v_proj",
        &normed,
        sequence_length,
        kv_dim,
        None,
        Some(&KvInvolvement {
            resource_id: kv_resource(layer, KvCacheRole::V),
            role: KvCacheRole::V,
            append: kv_append,
        }),
    );

    let rope_attrs = |head_count: u64| {
        vec![
            attr("base", AttributeValue::FloatValue(config.rope_theta)),
            attr("dimension", AttributeValue::Integer(head_dimension as i64)),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
            ),
            attr(
                "rotation_pairing",
                AttributeValue::Text(ROPE_ROTATION_PAIRING.to_string()),
            ),
            attr("position_offset", AttributeValue::Integer(0)),
            attr("head_count", AttributeValue::Integer(head_count as i64)),
        ]
    };
    let q_rope = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &rope_attrs(heads),
        &[q],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let k_rope = add_node(
        &format!("{prefix}.rope_k"),
        "rope",
        "PositionEncoding",
        &rope_attrs(kv_heads),
        &[k],
        &shape(vec![sequence_length, kv_dim]),
        Some(&KvInvolvement {
            resource_id: kv_resource(layer, KvCacheRole::K),
            role: KvCacheRole::K,
            append: kv_append,
        }),
    );

    let attn_out = add_node(
        &format!("{prefix}.attention"),
        "attention",
        "Attention",
        &[
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
        ],
        &[q_rope, k_rope, v],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
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
    let post_attn = add_node(
        &format!("{prefix}.residual1"),
        "residual-add",
        "Tensor",
        &[],
        &[residual_in.to_string(), attn_proj],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let mlp_normed = add_node(
        &format!("{prefix}.post_attn_norm"),
        "rmsnorm",
        "Normalization",
        &[epsilon()],
        &[
            post_attn.clone(),
            weight_edge(&weight_tensor_name(layer, "post_attn_norm")),
        ],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );
    let mlp_out = match moe {
        None => gated_mlp(
            &prefix,
            layer,
            "",
            &mlp_normed,
            sequence_length,
            hidden_size,
            config.intermediate_size,
        ),
        Some(moe) => moe_mlp(
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

/// DeepSeekMoE: `moe-experts(h, router(h), experts)` plus, when the
/// checkpoint has shared experts, their plain (ungated) gated MLP added to
/// the routed output.
fn moe_mlp(
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
    let routed = add_node(
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
    let config = model_config::model_config();
    let moe = model_config::mixture_of_experts();
    if let Some(reason) =
        unsupported_reason(&facts_of(&config, &model_config::options(), moe.as_ref()))
    {
        panic!("the DeepSeek Component cannot build this configuration: {reason}");
    }
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
        hidden = build_layer(
            &config,
            moe.as_ref(),
            layer,
            &hidden,
            sequence_length,
            kv_append,
        );
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

    fn supported() -> SupportFacts {
        SupportFacts {
            attention_bias: false,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            has_rope_scaling: false,
            has_layer_layout: false,
            has_final_logit_softcapping: false,
            has_per_layer_input: false,
            moe: Some(MoeFacts {
                experts: 64,
                experts_per_token: 6,
            }),
        }
    }

    #[test]
    fn names_its_architecture_family() {
        assert_eq!(ARCHITECTURE_FAMILY, "deepseek");
    }

    #[test]
    fn accepts_deepseek_moe_and_dense_shapes() {
        assert_eq!(unsupported_reason(&supported()), None);
        let dense = SupportFacts {
            moe: None,
            num_key_value_heads: 4,
            ..supported()
        };
        assert_eq!(unsupported_reason(&dense), None);
    }

    #[test]
    fn refuses_what_it_does_not_apply() {
        for facts in [
            SupportFacts {
                attention_bias: true,
                ..supported()
            },
            SupportFacts {
                has_rope_scaling: true,
                ..supported()
            },
            SupportFacts {
                has_layer_layout: true,
                ..supported()
            },
            SupportFacts {
                has_final_logit_softcapping: true,
                ..supported()
            },
            SupportFacts {
                has_per_layer_input: true,
                ..supported()
            },
            SupportFacts {
                num_key_value_heads: 5,
                ..supported()
            },
            SupportFacts {
                moe: Some(MoeFacts {
                    experts: 4,
                    experts_per_token: 5,
                }),
                ..supported()
            },
        ] {
            assert!(unsupported_reason(&facts).is_some(), "{facts:?}");
        }
    }

    #[test]
    fn weight_names_follow_the_canonical_layout() {
        assert_eq!(weight_tensor_name(3, "q_proj"), "layers.3.self_attn.q_proj");
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
