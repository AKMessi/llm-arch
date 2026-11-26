import { COMPONENT_MAP } from '../data/components';
import type { ArchitectureConfig, PhysicsReport } from '../types/architecture';
import { formatKiB, formatParams } from './physics';

export function generateHuggingFaceConfig(
  config: ArchitectureConfig,
  physics: PhysicsReport
): string {
  const { dims } = config;
  const tok = COMPONENT_MAP[config.tokenizer];
  const pos = COMPONENT_MAP[config.position];
  const norm = COMPONENT_MAP[config.norm];
  const attn = COMPONENT_MAP[config.attention];
  const hyb = COMPONENT_MAP[config.hybrid];
  const ffn = COMPONENT_MAP[config.ffn];
  const out = COMPONENT_MAP[config.output];

  const isMoE =
    config.ffn === 'ffn_moe_sparse' ||
    config.ffn === 'ffn_moe_shared' ||
    config.ffn === 'ffn_latent_moe' ||
    config.ffn === 'ffn_looped_moe' ||
    config.ffn === 'ffn_ngram_moe';
  const isMla =
    config.attention === 'attn_mla' ||
    config.attention === 'attn_mla_dsa' ||
    config.attention === 'attn_mla_dsa_loop' ||
    config.attention === 'attn_gated_mla' ||
    config.attention === 'attn_csa_hca';
  const isLoopHybridKv = config.attention === 'attn_mla_dsa_loop';
  const hasIndexShare =
    config.attention === 'attn_mla_dsa_loop' || config.attention === 'attn_gated_mla';
  const isSandwichLoop = config.ffn === 'ffn_looped_moe';
  const isFullLoop = config.ffn === 'ffn_looped_depth';
  const isLooped = isSandwichLoop || isFullLoop;

  const hf: Record<string, unknown> = {
    architectures: ['ArchitectonForCausalLM'],
    model_type: isSandwichLoop ? 'glm_loop_moe' : 'architecton_custom',
    _blueprint_name: config.name,
    _estimated_total_params: formatParams(physics.params.total),
    _estimated_active_params: formatParams(physics.params.active),
    _estimated_kv_cache_bf16_per_token: formatKiB(physics.kvKiBPerTokenBf16),
    vocab_size: physics.vocabSize,
    tokenizer_family: tok?.name,
    hidden_size: dims.dModel,
    num_hidden_layers: dims.nLayers,
    ...(isLooped ? { effective_num_hidden_layers: physics.effectiveDepth } : {}),
    num_attention_heads: dims.nHeads,
    num_key_value_heads:
      config.attention === 'attn_mha'
        ? dims.nHeads
        : config.attention === 'attn_mqa'
          ? 1
          : dims.nKvHeads,
    head_dim: physics.dHead,
    intermediate_size: dims.dFf,
    max_position_embeddings: dims.contextLength,
    position_encoding: pos?.id,
    rope_theta: dims.contextLength >= 262144 ? 10000000.0 : 500000.0,
    normalization_topology: norm?.id,
    gated_progressive_loop: config.norm === 'norm_gated_loop',
    qk_norm:
      config.norm === 'norm_rms_qknorm' ||
      config.norm === 'norm_peri_gemma' ||
      config.norm === 'norm_post_olmo',
    attention_mechanism: attn?.id,
    hybrid_interleave: hyb?.id,
    hybrid_ratio: config.hybrid !== 'hyb_none' ? `${dims.hybridRatio}:1` : null,
    sliding_window: config.hybrid === 'hyb_swa' || isLoopHybridKv ? dims.swaWindow : null,
    cross_loop_hybrid_kv: isLoopHybridKv,
    cross_loop_index_share: hasIndexShare,
    kv_lora_rank: isMla ? dims.mlaKvLoraRank : null,
    qk_rope_head_dim: isMla ? dims.mlaQkRopeDim : null,
    ffn_type: ffn?.id,
    num_routed_experts: isMoE ? dims.nExperts : null,
    num_experts_per_tok: isMoE ? dims.activeExperts : null,
    num_shared_experts:
      config.ffn === 'ffn_moe_shared' ||
      config.ffn === 'ffn_looped_moe' ||
      config.ffn === 'ffn_ngram_moe'
        ? dims.nSharedExperts
        : 0,
    ngram_engram_tables: config.ffn === 'ffn_ngram_moe',
    first_k_dense_replace: isMoE ? dims.densePrefixLayers : 0,
    looped_depth_passes: isLooped ? dims.loopPasses : 1,
    ...(physics.loopBreakdown
      ? {
          loop_layout: {
            opening_layers: physics.loopBreakdown.opening,
            looped_middle_layers: physics.loopBreakdown.looped,
            loop_passes: physics.loopBreakdown.passes,
            closing_layers: physics.loopBreakdown.closing,
          },
        }
      : {}),
    tie_word_embeddings: config.output === 'out_tied' || config.output === 'out_softcap',
    num_nextn_predict_layers:
      config.output === 'out_mtp' || config.output === 'out_sinks_mtp' ? 1 : 0,
    final_logit_softcapping: config.output === 'out_softcap' ? 30.0 : null,
    torch_dtype: dims.precision === 'fp32' ? 'float32' : 'bfloat16',
    _gallery_concepts: physics.conceptTags,
    _output_head: out?.id,
  };

  return JSON.stringify(hf, null, 2);
}

export function generatePyTorchCode(
  config: ArchitectureConfig,
  physics: PhysicsReport
): string {
  const { dims } = config;
  const tok = COMPONENT_MAP[config.tokenizer];
  const pos = COMPONENT_MAP[config.position];
  const norm = COMPONENT_MAP[config.norm];
  const attn = COMPONENT_MAP[config.attention];
  const hyb = COMPONENT_MAP[config.hybrid];
  const ffn = COMPONENT_MAP[config.ffn];
  const out = COMPONENT_MAP[config.output];

  const useRms = config.norm !== 'norm_pre_ln';
  const useQkNorm =
    config.norm === 'norm_rms_qknorm' ||
    config.norm === 'norm_peri_gemma' ||
    config.norm === 'norm_post_olmo';
  const isPeri = config.norm === 'norm_peri_gemma';
  const isPost = config.norm === 'norm_post_olmo';
  const isMhc = config.norm === 'norm_mhc';
  const isGatedLoop = config.norm === 'norm_gated_loop';
  const isMla =
    config.attention === 'attn_mla' ||
    config.attention === 'attn_mla_dsa' ||
    config.attention === 'attn_mla_dsa_loop' ||
    config.attention === 'attn_gated_mla' ||
    config.attention === 'attn_csa_hca';
  const isLoopHybridKv = config.attention === 'attn_mla_dsa_loop';
  const isGatedMla = config.attention === 'attn_gated_mla';
  const isGatedAttn =
    config.attention === 'attn_gated_gqa' || config.attention === 'attn_difftpa';
  const isMoE =
    config.ffn === 'ffn_moe_sparse' ||
    config.ffn === 'ffn_moe_shared' ||
    config.ffn === 'ffn_latent_moe' ||
    config.ffn === 'ffn_looped_moe' ||
    config.ffn === 'ffn_ngram_moe';
  const hasSharedExperts =
    (config.ffn === 'ffn_moe_shared' ||
      config.ffn === 'ffn_looped_moe' ||
      config.ffn === 'ffn_ngram_moe') &&
    dims.nSharedExperts > 0;
  const isSandwichLoop = config.ffn === 'ffn_looped_moe';
  const isFullLoop = config.ffn === 'ffn_looped_depth';
  const isTied = config.output === 'out_tied' || config.output === 'out_softcap';
  const hasMtp = config.output === 'out_mtp' || config.output === 'out_sinks_mtp';
  const hasSoftCap = config.output === 'out_softcap';
  const lb = physics.loopBreakdown;

  const normClass = useRms ? 'RMSNorm' : 'nn.LayerNorm';
  const nKv =
    config.attention === 'attn_mha'
      ? dims.nHeads
      : config.attention === 'attn_mqa'
        ? 1
        : dims.nKvHeads;

  return `"""
ARCHITECTON Generated Blueprint: ${config.name}
--------------------------------------------------------------------
Total Params : ${formatParams(physics.params.total)} (${formatParams(physics.params.active)} active · ${physics.params.activeRatioPct.toFixed(1)}%)
KV Cache     : ${formatKiB(physics.kvKiBPerTokenBf16)}/token (BF16) · ${physics.decoderTypeLabel}
Layer Mix    : ${physics.layerMixSummary}
Stack        : ${tok?.shortName} · ${pos?.shortName} · ${norm?.shortName} · ${attn?.shortName} · ${hyb?.shortName} · ${ffn?.shortName} · ${out?.shortName}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = ${physics.vocabSize}
    d_model: int = ${dims.dModel}
    n_layers: int = ${dims.nLayers}
    n_heads: int = ${dims.nHeads}
    n_kv_heads: int = ${nKv}
    d_head: int = ${physics.dHead}
    d_ff: int = ${dims.dFf}
    max_seq_len: int = ${dims.contextLength}${
      isMla
        ? `\n    kv_lora_rank: int = ${dims.mlaKvLoraRank}\n    qk_rope_dim: int = ${dims.mlaQkRopeDim}`
        : ''
    }${
      isMoE
        ? `\n    n_experts: int = ${dims.nExperts}\n    top_k_experts: int = ${dims.activeExperts}\n    n_shared_experts: int = ${hasSharedExperts ? dims.nSharedExperts : 0}\n    dense_prefix_layers: int = ${dims.densePrefixLayers}`
        : ''
    }${
      config.hybrid === 'hyb_swa' || isLoopHybridKv
        ? `\n    sliding_window: int = ${dims.swaWindow}`
        : ''
    }${
      config.hybrid !== 'hyb_none' ? `\n    hybrid_ratio: int = ${dims.hybridRatio}` : ''
    }${
      isFullLoop
        ? `\n    loop_passes: int = ${dims.loopPasses}`
        : isSandwichLoop && lb
          ? `\n    loop_opening_layers: int = ${lb.opening}\n    loop_middle_layers: int = ${lb.looped}\n    loop_closing_layers: int = ${lb.closing}\n    loop_passes: int = ${lb.passes}  # Effective depth = ${physics.effectiveDepth} layers`
          : ''
    }


${
  useRms
    ? `class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight`
    : `# Using torch.nn.LayerNorm (Pre-LN GPT-2 style)`
}


${
  isMla
    ? `class MultiHeadLatentAttention(nn.Module):
    """${attn?.name} — compresses K/V jointly into low-rank latent c_KV (${dims.mlaKvLoraRank}d).${
      isLoopHybridKv
        ? `\n    Cross-Loop Hybrid KV: Pass 1 caches global (c_KV, k_rope) + DSA indices; Pass 2 reuses both + local SWA(${dims.swaWindow}).`
        : ''
    }"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * (cfg.d_head + cfg.qk_rope_dim), bias=False)
        self.w_dkv = nn.Linear(cfg.d_model, cfg.kv_lora_rank + cfg.qk_rope_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.w_ukv = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * 2 * cfg.d_head, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)${
          isGatedMla
            ? `\n        self.w_gate = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)  # Gated MLA output gate`
            : ''
        }

    def forward(self, x: torch.Tensor${isLoopHybridKv ? ', loop_pass: int = 0' : ''}) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, -1)
        compressed = self.w_dkv(x)
        c_kv, k_rope = compressed.split([compressed.size(-1) - ${dims.mlaQkRopeDim}, ${dims.mlaQkRopeDim}], dim=-1)
        kv = self.w_ukv(self.kv_norm(c_kv)).view(B, T, self.n_heads, 2 * self.d_head)
        k_nope, v = kv.chunk(2, dim=-1)
        # Note: At inference, cache ONLY (c_kv, k_rope) → ${formatKiB(physics.kvKiBPerTokenBf16)}/tok!${
          isLoopHybridKv
            ? `\n        # On loop_pass > 0, production GLM-5.4 reuses Pass-1 global c_kv + DSA sparse indices (IndexShare)\n        # and keeps only a local sliding-window KV ring (W=${dims.swaWindow}) for refreshed tokens!`
            : ''
        }
        k = torch.cat([k_nope, k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)${
          isGatedMla ? `\n        out = out * torch.sigmoid(self.w_gate(x))  # Sink-free Gated MLA` : ''
        }
        return self.w_o(out)`
    : `class SequenceAttention(nn.Module):
    """${attn?.name} (${dims.nHeads} Q heads : ${nKv} KV heads${useQkNorm ? ' + QK-Norm' : ''}${isGatedAttn ? ' + Output Gate σ' : ''})."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.d_head = cfg.n_heads, cfg.n_kv_heads, cfg.d_head
        self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)${
          useQkNorm
            ? `\n        self.q_norm = ${normClass}(cfg.d_head)\n        self.k_norm = ${normClass}(cfg.d_head)`
            : ''
        }${
          isGatedAttn
            ? `\n        self.w_gate = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)`
            : ''
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.w_k(x).view(B, T, self.n_kv, self.d_head)
        v = self.w_v(x).view(B, T, self.n_kv, self.d_head)${
          useQkNorm ? `\n        q, k = self.q_norm(q), self.k_norm(k)` : ''
        }
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)${
          isGatedAttn ? `\n        out = out * torch.sigmoid(self.w_gate(x))  # Sink-free output gate` : ''
        }
        return self.w_o(out)`
}


${
  isMoE
    ? `class SwiGLUExpert(nn.Module):
    def __init__(self, d_in: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_ff, bias=False)
        self.w_up = nn.Linear(d_in, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SparseMoEBlock(nn.Module):
    """${ffn?.name}: ${dims.nExperts} routed experts (top-${dims.activeExperts} active${hasSharedExperts ? ` + ${dims.nSharedExperts} shared` : ''})."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.top_k = cfg.top_k_experts${
          config.ffn === 'ffn_latent_moe'
            ? `\n        self.d_lat = cfg.d_model // 2\n        self.lat_down = nn.Linear(cfg.d_model, self.d_lat, bias=False)\n        self.lat_up = nn.Linear(self.d_lat, cfg.d_model, bias=False)`
            : ''
        }
        d_exp = ${config.ffn === 'ffn_latent_moe' ? 'cfg.d_model // 2' : 'cfg.d_model'}
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_exp, cfg.d_ff) for _ in range(cfg.n_experts)])${
          hasSharedExperts
            ? `\n        self.shared = SwiGLUExpert(cfg.d_model, cfg.d_ff * cfg.n_shared_experts)`
            : ''
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.router(x), dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        z = ${config.ffn === 'ffn_latent_moe' ? 'self.lat_down(x)' : 'x'}
        out = torch.zeros_like(z)
        for k in range(self.top_k):
            # Reference dispatch loop (production uses grouped GEMM / MegaBlocks)
            idx_k, w_k = indices[..., k], weights[..., k].unsqueeze(-1)
            for e in idx_k.unique().tolist():
                mask = idx_k == e
                out[mask] += w_k[mask] * self.experts[e](z[mask])${
                  config.ffn === 'ffn_latent_moe' ? `\n        out = self.lat_up(out)` : ''
                }${
                  hasSharedExperts
                    ? `\n        out = out + self.shared(x)`
                    : ''
                }
        return out`
    : `class DenseFFN(nn.Module):
    """${ffn?.name} (d_model=${dims.dModel} → d_ff=${dims.dFf})."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)${
          config.ffn !== 'ffn_gelu' ? `\n        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)` : ''
        }
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:${
      config.ffn === 'ffn_gelu'
        ? `\n        return self.w_down(F.gelu(self.w_up(x), approximate="tanh"))`
        : config.ffn === 'ffn_geglu'
          ? `\n        return self.w_down(F.gelu(self.w_gate(x), approximate="tanh") * self.w_up(x))`
          : `\n        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))`
    }`
}


class DecoderBlock(nn.Module):
    """Topology: ${norm?.name} · Mixer: ${attn?.shortName}${config.hybrid !== 'hyb_none' ? ` / ${hyb?.shortName}` : ''}"""
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = ${normClass}(cfg.d_model)
        self.norm2 = ${normClass}(cfg.d_model)${
          isPeri
            ? `\n        self.post_attn_norm = ${normClass}(cfg.d_model)\n        self.post_ffn_norm = ${normClass}(cfg.d_model)`
            : ''
        }
        self.mixer = ${isMla ? 'MultiHeadLatentAttention(cfg)' : 'SequenceAttention(cfg)'}
        self.ffn = ${
          isMoE
            ? 'SwiGLUExpert(cfg.d_model, max(cfg.d_ff * 4, int(cfg.d_model * 2.7))) if layer_idx < cfg.dense_prefix_layers else SparseMoEBlock(cfg)'
            : 'DenseFFN(cfg)'
        }

    def forward(self, x: torch.Tensor${isLoopHybridKv ? ', loop_pass: int = 0' : ''}) -> torch.Tensor:${
      isPeri
        ? `\n        x = x + self.post_attn_norm(self.mixer(self.norm1(x)${isLoopHybridKv ? ', loop_pass=loop_pass' : ''}))\n        x = x + self.post_ffn_norm(self.ffn(self.norm2(x)))\n        return x`
        : isPost
          ? `\n        x = x + self.norm1(self.mixer(x${isLoopHybridKv ? ', loop_pass=loop_pass' : ''}))\n        x = x + self.norm2(self.ffn(x))\n        return x`
          : isMhc
            ? `\n        # mHC: Manifold-Constrained Hyper-Connection multi-stream residual update\n        x = x + self.mixer(self.norm1(x)${isLoopHybridKv ? ', loop_pass=loop_pass' : ''})\n        x = x + self.ffn(self.norm2(x))\n        return x`
            : `\n        x = x + self.mixer(self.norm1(x)${isLoopHybridKv ? ', loop_pass=loop_pass' : ''})\n        x = x + self.ffn(self.norm2(x))\n        return x`
    }


class ArchitectonLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)${
          config.position === 'pos_abs_learned'
            ? `\n        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)`
            : ''
        }${
          isSandwichLoop && lb
            ? `\n        # GLM-5.4 Sandwich Loop Layout: ${lb.opening} opening + ${lb.looped} shared looped (×${lb.passes}) + ${lb.closing} closing = ${physics.effectiveDepth} eff. layers\n        self.opening_layers = nn.ModuleList([DecoderBlock(cfg, i) for i in range(cfg.loop_opening_layers)])\n        self.loop_layers = nn.ModuleList([DecoderBlock(cfg, cfg.loop_opening_layers + i) for i in range(cfg.loop_middle_layers)])\n        self.closing_layers = nn.ModuleList([DecoderBlock(cfg, cfg.loop_opening_layers + cfg.loop_middle_layers + i) for i in range(cfg.loop_closing_layers)])`
            : `\n        self.layers = nn.ModuleList([DecoderBlock(cfg, i) for i in range(cfg.n_layers)])`
        }${
          isGatedLoop
            ? `\n        # Independent per-token progressive iteration gates β_k (prevents loop norm blowup)\n        self.loop_gates = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=True) for _ in range(${Math.max(2, dims.loopPasses)})])`
            : ''
        }
        self.final_norm = ${normClass}(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)${
          isTied ? `\n        self.lm_head.weight = self.tok_emb.weight  # Weight-tied embeddings` : ''
        }${
          hasMtp
            ? `\n        self.mtp_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)  # MTP-1 head`
            : ''
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)${
          config.position === 'pos_abs_learned'
            ? `\n        x = x + self.pos_emb(torch.arange(input_ids.size(1), device=input_ids.device))`
            : ''
        }${
          isSandwichLoop && lb
            ? `\n        for layer in self.opening_layers:\n            x = layer(x${isLoopHybridKv ? ', loop_pass=0' : ''})\n        for k in range(self.cfg.loop_passes):\n            h_in = x\n            for layer in self.loop_layers:\n                x = layer(x${isLoopHybridKv ? ', loop_pass=k' : ''})${
                isGatedLoop
                  ? `\n            beta_k = torch.sigmoid(self.loop_gates[k](h_in))\n            x = h_in + beta_k * (x - h_in)  # Gated progressive update β_k`
                  : ''
              }\n        for layer in self.closing_layers:\n            x = layer(x${isLoopHybridKv ? ', loop_pass=0' : ''})`
            : isFullLoop
              ? `\n        for k in range(self.cfg.loop_passes):\n            h_in = x\n            for layer in self.layers:\n                x = layer(x${isLoopHybridKv ? ', loop_pass=k' : ''})${
                  isGatedLoop
                    ? `\n            beta_k = torch.sigmoid(self.loop_gates[k](h_in))\n            x = h_in + beta_k * (x - h_in)`
                    : ''
                }`
              : `\n        for layer in self.layers:\n            x = layer(x)`
        }
        logits = self.lm_head(self.final_norm(x))${
          hasSoftCap ? `\n        logits = 30.0 * torch.tanh(logits / 30.0)  # Gemma logit soft-capping` : ''
        }
        return logits
`;
}

export function encodeConfigToHash(config: ArchitectureConfig): string {
  try {
    const json = JSON.stringify(config);
    return btoa(unescape(encodeURIComponent(json)));
  } catch {
    return '';
  }
}

export function decodeConfigFromHash(hash: string): ArchitectureConfig | null {
  try {
    const cleaned = hash.replace(/^#bp=/, '');
    if (!cleaned) return null;
    const json = decodeURIComponent(escape(atob(cleaned)));
    const parsed = JSON.parse(json) as ArchitectureConfig;
    if (parsed && parsed.tokenizer && parsed.attention && parsed.dims) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}
