import type { ArchitectureConfig, DiagnosticItem, PhysicsReport } from '../types/architecture';

export function runArchitectureDiagnostics(
  config: ArchitectureConfig,
  physics: PhysicsReport
): DiagnosticItem[] {
  const items: DiagnosticItem[] = [];
  const { dims } = config;

  // 1. Absolute Pos + Long Context critical incompatibility
  if (config.position === 'pos_abs_learned' && dims.contextLength > 4096) {
    items.push({
      id: 'abs-pos-long-ctx',
      severity: 'critical',
      title: 'Absolute Positional Ceiling Exceeded',
      detail: `Learned Absolute Embeddings cannot extrapolate reliably to ${Math.round(dims.contextLength / 1024)}K context and bloat parameters by ${((dims.contextLength * dims.dModel) / 1e6).toFixed(0)}M. Switch to YaRN RoPE or Local-RoPE + Global-NoPE.`,
      suggestion: {
        label: 'Snap YaRN RoPE',
        stage: 'position',
        componentId: 'pos_yarn_rope',
      },
    });
  }

  // 2. Massive KV-Cache Warning on MHA at long context
  if (config.attention === 'attn_mha' && config.hybrid === 'hyb_none' && dims.contextLength >= 65536) {
    items.push({
      id: 'mha-kv-explosion',
      severity: 'warning',
      title: `KV-Cache Memory Wall (${physics.kvCacheGbAtContext.toFixed(1)} GB)`,
      detail: `Uniform MHA requires ${physics.kvKiBPerTokenBf16.toFixed(0)} KiB/tok BF16 — starving GPU VRAM at ${Math.round(dims.contextLength / 1024)}K context. Drop MLA or a Gated DeltaNet hybrid to cut KV memory by 5×–14×.`,
      suggestion: {
        label: 'Swap to MLA',
        stage: 'attention',
        componentId: 'attn_mla',
      },
    });
  }

  // 3. Large Vocab + Untied Embeddings on Small Model
  const embedShare = ((physics.params.embeddings + physics.params.outputHead) / Math.max(1, physics.params.total)) * 100;
  if (
    physics.vocabSize >= 200000 &&
    dims.dModel <= 2560 &&
    config.output === 'out_untied' &&
    embedShare > 28
  ) {
    items.push({
      id: 'wide-vocab-untied',
      severity: 'warning',
      title: `Embeddings Consume ${embedShare.toFixed(0)}% of Total Weights`,
      detail: `Pairing a ${(physics.vocabSize / 1000).toFixed(0)}k vocabulary with an untied LM head on d_model=${dims.dModel} spends massive capacity on lookup tables. Weight-Tying immediately reclaims ${((physics.vocabSize * dims.dModel) / 1e6).toFixed(0)}M params.`,
      suggestion: {
        label: 'Snap Weight-Tied Head',
        stage: 'output',
        componentId: 'out_tied',
      },
    });
  }

  // 4. Local-RoPE + Global-NoPE without a local/SWA hybrid stack
  if (config.position === 'pos_local_rope_global_nope' && config.hybrid === 'hyb_none') {
    items.push({
      id: 'local-rope-needs-hybrid',
      severity: 'info',
      title: 'Local-RoPE + Global-NoPE Pairs Best with Interleaved SWA',
      detail: `Models like Muse Spark, Gemma 4, and Command A+ interleave 3:1 or 5:1 Sliding-Window layers (with RoPE) and Global layers (with NoPE). Drop SWA Hybrid to unlock this exact pattern.`,
      suggestion: {
        label: 'Add SWA : Global Mix',
        stage: 'hybrid',
        componentId: 'hyb_swa',
      },
    });
  }

  // 5. Pure NoPE without Recurrence / ShortConv
  if (config.position === 'pos_pure_nope' && (config.hybrid === 'hyb_none' || config.hybrid === 'hyb_swa')) {
    items.push({
      id: 'pure-nope-no-conv',
      severity: 'warning',
      title: 'Pure NoPE Without ShortConv / SSM Layers',
      detail: `Frontier NoPE architectures (Ember-1, Solar Open 2, Nemotron 3) pair NoPE global attention with Mamba-2 or Kimi Delta Attention + ShortConv to maintain crisp local word-order sensitivity.`,
      suggestion: {
        label: 'Add Kimi KDA Hybrid',
        stage: 'hybrid',
        componentId: 'hyb_kda',
      },
    });
  }

  // 6. High-scale deep stack without QK-Norm or mHC
  if (
    dims.nLayers >= 48 &&
    (config.norm === 'norm_pre_ln' || config.norm === 'norm_pre_rms') &&
    config.attention !== 'attn_mla' &&
    config.attention !== 'attn_mla_dsa' &&
    config.attention !== 'attn_csa_hca'
  ) {
    items.push({
      id: 'deep-stack-stability',
      severity: 'info',
      title: 'Consider QK-Norm for Deep-Stack Stability',
      detail: `At ${dims.nLayers} layers, QK-Norm (used in Qwen3, Gemma 3/4, OLMo 3, GLM-4.5) prevents attention logit entropy collapse and enables higher learning rates & FP8 training.`,
      suggestion: {
        label: 'Upgrade to QK-Norm',
        stage: 'norm',
        componentId: 'norm_rms_qknorm',
      },
    });
  }

  // ==================== SYNERGIES (POSITIVE ARCHITECTURAL INSIGHTS) ====================
  if (config.attention === 'attn_gated_gqa' && (config.hybrid === 'hyb_deltanet' || config.hybrid === 'hyb_kda')) {
    items.push({
      id: 'syn-qwen-hybrid',
      severity: 'synergy',
      title: 'Frontier Hybrid Synergy: Linear Recurrence + Gated Attention',
      detail: `Matching Qwen3.5/3.8 & Ember-1! ${dims.hybridRatio}:1 linear recurrence eliminates ${Math.round((dims.hybridRatio / (dims.hybridRatio + 1)) * 100)}% of KV-cache while Gated Attention preserves exact needle-in-a-haystack recall without attention sinks.`,
    });
  }

  if (
    (config.attention === 'attn_mla' || config.attention === 'attn_mla_dsa' || config.attention === 'attn_csa_hca') &&
    (config.ffn === 'ffn_moe_shared' || config.ffn === 'ffn_latent_moe')
  ) {
    items.push({
      id: 'syn-deepseek-frontier',
      severity: 'synergy',
      title: 'DeepSeek-Class Memory + Compute Frontier',
      detail: `Combining latent KV compression (${physics.kvKiBPerTokenBf16.toFixed(1)} KiB/tok) with ${physics.params.activeRatioPct.toFixed(1)}% active sparse MoE unlocks massive batch throughput even at 128K–1M context.`,
    });
  }

  if (config.hybrid === 'hyb_mamba2' || config.hybrid === 'hyb_xlstm') {
    items.push({
      id: 'syn-ssm-kv',
      severity: 'synergy',
      title:
        config.hybrid === 'hyb_xlstm'
          ? 'Zero-KV Constant-Memory Recurrence'
          : `Ultra-Lean SSM Hybrid (${physics.kvKiBPerTokenBf16.toFixed(1)} KiB/tok)`,
      detail:
        config.hybrid === 'hyb_xlstm'
          ? `0 B/token KV-cache! Your memory footprint stays completely flat at ${physics.recurrentStateMb.toFixed(0)} MB whether at 4K or 1M tokens.`
          : `Mamba-2 SSM layers handle sequential compression in O(1) memory while periodic attention layers guarantee long-range associative recall — matching NVIDIA Nemotron 3.`,
    });
  }

  if (config.output === 'out_mtp' || config.output === 'out_sinks_mtp') {
    items.push({
      id: 'syn-mtp',
      severity: 'synergy',
      title: 'Native Speculative Decoding Ready (MTP)',
      detail: `Multi-Token Prediction enriches training gradients AND doubles as a zero-overhead draft head at inference for ~1.65× wall-clock decode speedup.`,
    });
  }

  if (config.norm === 'norm_mhc') {
    items.push({
      id: 'syn-mhc',
      severity: 'synergy',
      title: 'mHC Multi-Stream Residual Highway Active',
      detail: `Doubly-stochastic Birkhoff mixing across 4 parallel residual streams enhances representational capacity without exploding activation norms (DeepSeek V4 / ERNIE 5.5 pattern).`,
    });
  }

  // Fallback positive note if completely clean
  if (items.length === 0) {
    items.push({
      id: 'clean-balanced',
      severity: 'synergy',
      title: 'Clean, Production-Balanced Decoder Stack',
      detail: `All 7 architectural stages align harmoniously with zero structural conflicts and ${physics.kvCompressionVsMha.toFixed(1)}× KV compression vs baseline MHA.`,
    });
  }

  return items;
}
