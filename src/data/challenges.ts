import type {
  ArchitectureConfig,
  DiagnosticItem,
  PhysicsReport,
} from '../types/architecture';
import { PRESET_BLUEPRINTS } from './presets';

export interface ChallengeConstraint {
  id: string;
  label: string;
  currentText: string;
  passed: boolean;
}

export interface ArchitectureChallenge {
  id: string;
  number: string;
  title: string;
  difficulty: 'Starter' | 'Engineer' | 'Principal' | 'Frontier';
  category: string;
  brief: string;
  hint: string;
  realWorldRef: string;
  starterPresetId: string;
  evaluate: (
    config: ArchitectureConfig,
    physics: PhysicsReport,
    diagnostics: DiagnosticItem[]
  ) => ChallengeConstraint[];
}

export interface EvaluatedChallenge {
  challenge: ArchitectureChallenge;
  constraints: ChallengeConstraint[];
  passedCount: number;
  totalCount: number;
  completed: boolean;
}

export const ARCHITECTURE_CHALLENGES: ArchitectureChallenge[] = [
  {
    id: 'rtx-4090-sweetspot',
    number: '01',
    title: 'The Single RTX 4090 Monster',
    difficulty: 'Starter',
    category: 'Consumer Edge',
    brief:
      'Engineer a ≥12B total-parameter model with a 128K+ context window that STILL fits cleanly inside a single 24 GB RTX 4090 with blazing speed.',
    hint: 'Switch precision to INT4 or FP8, drop in Sparse MoE or MLA / Hybrid layers to shrink KV-cache, and balance d_model / layers!',
    realWorldRef: 'Inspired by Qwen3 30B-A3B, Gemma 4 26B-A4B & Nemotron 3 Nano',
    starterPresetId: 'llama-3-8b',
    evaluate: (config, physics) => [
      {
        id: 'total-12b',
        label: 'Total Parameters ≥ 12B',
        currentText: `${(physics.params.total / 1e9).toFixed(1)}B`,
        passed: physics.params.total >= 12e9,
      },
      {
        id: 'ctx-128k',
        label: 'Context Window ≥ 128K',
        currentText: `${Math.round(config.dims.contextLength / 1024)}K`,
        passed: config.dims.contextLength >= 131072,
      },
      {
        id: 'fits-4090',
        label: 'Fits on 1× RTX 4090 (≤ 22 GB VRAM)',
        currentText: `${physics.totalVramGb.toFixed(1)} GB`,
        passed: physics.totalVramGb <= 22.08,
      },
      {
        id: 'fast-decode',
        label: 'Est. H100 Decode ≥ 140 tok/s',
        currentText: `${Math.round(physics.estDecodeTokSecH100)} tok/s`,
        passed: physics.estDecodeTokSecH100 >= 140,
      },
    ],
  },
  {
    id: 'million-token-survivor',
    number: '02',
    title: 'Million-Token Context Survivor',
    difficulty: 'Engineer',
    category: 'Long Context',
    brief:
      'Build a ≥30B total-parameter architecture capable of a full 1,048,576 (1M) token context while keeping BF16 KV-cache ≤ 24 KiB/token and zero critical warnings.',
    hint: 'Combine Mamba-2 SSM, Gated DeltaNet, or DeepSeek V4 CSA/HCA with YaRN RoPE or Local-RoPE + Global-NoPE!',
    realWorldRef: 'Inspired by Nemotron 3 Super (8 KiB/tok) & DeepSeek V4 (7.7 KiB/tok)',
    starterPresetId: 'qwen3-32b',
    evaluate: (config, physics, diagnostics) => {
      const hasCritical = diagnostics.some((d) => d.severity === 'critical');
      return [
        {
          id: 'ctx-1m',
          label: 'Context Window = 1M tokens',
          currentText: `${Math.round(config.dims.contextLength / 1024)}K`,
          passed: config.dims.contextLength >= 1048576,
        },
        {
          id: 'kv-24',
          label: 'KV-Cache ≤ 24 KiB/token (BF16)',
          currentText: `${physics.kvKiBPerTokenBf16.toFixed(1)} KiB`,
          passed: physics.kvKiBPerTokenBf16 <= 24,
        },
        {
          id: 'params-30b',
          label: 'Total Parameters ≥ 30B',
          currentText: `${(physics.params.total / 1e9).toFixed(1)}B`,
          passed: physics.params.total >= 30e9,
        },
        {
          id: 'no-crit',
          label: 'Zero Critical Advisor Warnings',
          currentText: hasCritical ? 'Critical Issue' : 'Clean',
          passed: !hasCritical,
        },
      ];
    },
  },
  {
    id: 'time-machine-2019-2026',
    number: '03',
    title: '2019 → 2026 Modernization',
    difficulty: 'Starter',
    category: 'Architecture Evolution',
    brief:
      'Start from GPT-2 XL (2019: 300 KiB/tok KV, 1K context, Pre-LN, MHA) and modernize every weak link to reach ≥128K context and ≥10× KV compression with zero warnings.',
    hint: 'Upgrade Position (RoPE/YaRN), Norm (RMSNorm/QK-Norm), Attention (MLA/GQA), FFN (SwiGLU/MoE), and extend Context to ≥128K.',
    realWorldRef: 'Bridging 7 years of open-weight LLM innovation from GPT-2 to 2026',
    starterPresetId: 'gpt-2-xl',
    evaluate: (config, physics, diagnostics) => {
      const noWarnings = !diagnostics.some((d) => d.severity === 'critical' || d.severity === 'warning');
      return [
        {
          id: 'ctx-128k',
          label: 'Context ≥ 128K (up from 1K)',
          currentText: `${Math.round(config.dims.contextLength / 1024)}K`,
          passed: config.dims.contextLength >= 131072,
        },
        {
          id: 'kv-cut',
          label: 'KV Compression ≥ 10× vs MHA',
          currentText: physics.kvCompressionVsMha >= 999 ? '∞×' : `${physics.kvCompressionVsMha.toFixed(1)}×`,
          passed: physics.kvCompressionVsMha >= 10,
        },
        {
          id: 'modern-norm',
          label: 'Modern RMSNorm / QK-Norm / mHC',
          currentText: config.norm === 'norm_pre_ln' ? 'Pre-LN (Legacy)' : 'Modernized',
          passed: config.norm !== 'norm_pre_ln',
        },
        {
          id: 'clean-advisor',
          label: 'Zero Warnings or Critical Issues',
          currentText: noWarnings ? 'Clean' : 'Has Warnings',
          passed: noWarnings,
        },
      ];
    },
  },
  {
    id: 'frontier-5pct-titan',
    number: '04',
    title: 'The 5%-Active Frontier Titan',
    difficulty: 'Principal',
    category: 'Sparse MoE',
    brief:
      'Construct a trillion-class or frontier MoE (≥250B total) where ≤6.5% of parameters activate per token, paired with MTP speculative decoding and ≤75 KiB/tok KV.',
    hint: 'Equip Fine-Grained MoE + Shared Experts or Latent MoE, scale to 256+ experts with top-6/8 active, drop MLA or Hybrid mixer, and add MTP heads!',
    realWorldRef: 'Inspired by DeepSeek V3 (5.5% active), Qwen3.5 (4.3%) & GLM-5 (5.4%)',
    starterPresetId: 'llama-3-8b',
    evaluate: (config, physics) => [
      {
        id: 'tot-250b',
        label: 'Total Parameters ≥ 250B',
        currentText: `${(physics.params.total / 1e9).toFixed(0)}B`,
        passed: physics.params.total >= 250e9,
      },
      {
        id: 'act-ratio',
        label: 'Active Parameter Ratio ≤ 6.5%',
        currentText: `${physics.params.activeRatioPct.toFixed(1)}%`,
        passed: physics.params.activeRatioPct <= 6.5,
      },
      {
        id: 'kv-75',
        label: 'KV-Cache ≤ 75 KiB/tok (BF16)',
        currentText: `${physics.kvKiBPerTokenBf16.toFixed(1)} KiB`,
        passed: physics.kvKiBPerTokenBf16 <= 75,
      },
      {
        id: 'has-mtp',
        label: 'Multi-Token Prediction (MTP) Enabled',
        currentText: config.output === 'out_mtp' || config.output === 'out_sinks_mtp' ? 'Active' : 'Missing',
        passed: config.output === 'out_mtp' || config.output === 'out_sinks_mtp',
      },
    ],
  },
  {
    id: 'sink-slayer-dense',
    number: '05',
    title: 'The Sink-Free Deep Dense Stack',
    difficulty: 'Engineer',
    category: 'Training Stability',
    brief:
      'Design a rock-solid ≥48-layer Dense (100% active) transformer with QK-Norm spike suppression, Sink-Free Gated GQA, and interleaved SWA or DeltaNet hybrid efficiency.',
    hint: 'Equip Pre-RMSNorm + QK-Norm (or Peri-Norm), Gated GQA, SWA or Gated DeltaNet hybrid, Dense SwiGLU/GeGLU, and ≥48 layers.',
    realWorldRef: 'Inspired by Qwen3.8 27B & Meta Muse Glimmer 30B',
    starterPresetId: 'qwen3-32b',
    evaluate: (config, physics) => {
      const hasQkNorm =
        config.norm === 'norm_rms_qknorm' ||
        config.norm === 'norm_peri_gemma' ||
        config.norm === 'norm_post_olmo';
      const isDense = physics.params.activeRatioPct >= 99;
      return [
        {
          id: 'layers-48',
          label: 'Decoder Depth ≥ 48 Layers',
          currentText: `${config.dims.nLayers} layers`,
          passed: config.dims.nLayers >= 48,
        },
        {
          id: 'dense-100',
          label: 'Dense Stack (100% Active)',
          currentText: `${physics.params.activeRatioPct.toFixed(0)}%`,
          passed: isDense,
        },
        {
          id: 'qk-norm',
          label: 'QK-Norm Spike Protection',
          currentText: hasQkNorm ? 'Enabled' : 'Missing',
          passed: hasQkNorm,
        },
        {
          id: 'gated-hybrid',
          label: 'Gated GQA + Hybrid Interleave',
          currentText:
            config.attention === 'attn_gated_gqa' && config.hybrid !== 'hyb_none'
              ? 'Active'
              : 'Incomplete',
          passed: config.attention === 'attn_gated_gqa' && config.hybrid !== 'hyb_none',
        },
      ];
    },
  },
  {
    id: 'polyglot-edge-diet',
    number: '06',
    title: 'The 262k Polyglot Edge Diet',
    difficulty: 'Engineer',
    category: 'Tokenization & Compression',
    brief:
      'Equip an ultra-wide ≥248k multilingual vocabulary on a compact ≤5B parameter edge model while keeping Embeddings + LM Head under 25% of total weights.',
    hint: 'Drop SentencePiece 262k or Wide BPE 248k, dial d_model ≤ 2048, and snap in Weight-Tied LM Head (or Tied + SoftCap) to cut embedding params in half!',
    realWorldRef: 'Inspired by Gemma 3 1B/4B & Gemma 4 E2B',
    starterPresetId: 'gemma-4-e4b',
    evaluate: (config, physics) => {
      const embShare =
        ((physics.params.embeddings + physics.params.outputHead) / Math.max(1, physics.params.total)) * 100;
      const isTied = config.output === 'out_tied' || config.output === 'out_softcap';
      return [
        {
          id: 'vocab-248k',
          label: 'Vocabulary ≥ 248,000 tokens',
          currentText: `${(physics.vocabSize / 1000).toFixed(0)}k`,
          passed: physics.vocabSize >= 248000,
        },
        {
          id: 'small-5b',
          label: 'Compact Footprint ≤ 5.5B Total',
          currentText: `${(physics.params.total / 1e9).toFixed(2)}B`,
          passed: physics.params.total <= 5.5e9,
        },
        {
          id: 'emb-budget',
          label: 'Embed + Head Share ≤ 26% of Weights',
          currentText: `${embShare.toFixed(1)}%`,
          passed: embShare <= 26,
        },
        {
          id: 'tied-head',
          label: 'Weight-Tied Embeddings Active',
          currentText: isTied ? 'Tied' : 'Untied',
          passed: isTied,
        },
      ];
    },
  },
  {
    id: 'deep-thinker-loop',
    number: '07',
    title: 'Recurrent-Depth Reasoning Lab',
    difficulty: 'Principal',
    category: '2026 Frontier',
    brief:
      'Achieve ≥96 effective reasoning layers of depth while storing ≤8B physical parameters in VRAM!',
    hint: 'Drop Looped Depth Sharing (×K passes) or Looped Sparse MoE into Stage 06 and tune physical layers + loop passes (×2 or ×4) inside Fine-Tune!',
    realWorldRef: 'Inspired by ByteDance Ouro-Thinking 2.6B (48×4), NVIDIA Dory 1.0 & Z.ai GLM-5.4 (26+28×2+24)',
    starterPresetId: 'llama-3-8b',
    evaluate: (config, physics) => [
      {
        id: 'eff-depth',
        label: 'Effective Depth ≥ 96 Layers',
        currentText: `${physics.effectiveDepth} eff. layers`,
        passed: physics.effectiveDepth >= 96,
      },
      {
        id: 'phys-8b',
        label: 'Physical Parameters ≤ 8.5B',
        currentText: `${(physics.params.total / 1e9).toFixed(2)}B`,
        passed: physics.params.total <= 8.5e9,
      },
      {
        id: 'looped',
        label: 'Looped Depth Sharing Equipped',
        currentText:
          config.ffn === 'ffn_looped_depth' || config.ffn === 'ffn_looped_moe'
            ? `×${config.dims.loopPasses} passes`
            : 'Not Equipped',
        passed: config.ffn === 'ffn_looped_depth' || config.ffn === 'ffn_looped_moe',
      },
    ],
  },
  {
    id: 'zero-kv-infinite',
    number: '08',
    title: 'Zero-KV Infinite Streamer',
    difficulty: 'Frontier',
    category: 'Post-Transformer',
    brief:
      'Break free from quadratic KV-cache growth completely: build a ≥7B architecture with literally 0 B/token KV-cache supporting 100+ concurrent 256K streams on 1× H100!',
    hint: 'Drop Pure xLSTM / mLSTM Recurrent into Stage 05 Hybrid Stack, pair with Pure NoPE, and set context ≥ 256K.',
    realWorldRef: 'Inspired by NXAI xLSTM 7B (0 B/tok KV)',
    starterPresetId: 'llama-3-8b',
    evaluate: (config, physics) => [
      {
        id: 'zero-kv',
        label: 'KV-Cache = 0 B / token',
        currentText: `${physics.kvKiBPerTokenBf16.toFixed(1)} KiB`,
        passed: physics.kvKiBPerTokenBf16 === 0,
      },
      {
        id: 'size-7b',
        label: 'Total Parameters ≥ 7B',
        currentText: `${(physics.params.total / 1e9).toFixed(1)}B`,
        passed: physics.params.total >= 7e9,
      },
      {
        id: 'ctx-256k',
        label: 'Context ≥ 256K tokens',
        currentText: `${Math.round(config.dims.contextLength / 1024)}K`,
        passed: config.dims.contextLength >= 262144,
      },
      {
        id: 'conc-100',
        label: '≥ 50 Concurrent Users on 1× H100',
        currentText: `${physics.economics.maxConcurrent1xH100} users`,
        passed: physics.economics.maxConcurrent1xH100 >= 50,
      },
    ],
  },
];

export function evaluateAllChallenges(
  config: ArchitectureConfig,
  physics: PhysicsReport,
  diagnostics: DiagnosticItem[]
): EvaluatedChallenge[] {
  return ARCHITECTURE_CHALLENGES.map((challenge) => {
    const constraints = challenge.evaluate(config, physics, diagnostics);
    const passedCount = constraints.filter((c) => c.passed).length;
    return {
      challenge,
      constraints,
      passedCount,
      totalCount: constraints.length,
      completed: passedCount === constraints.length,
    };
  });
}

export function getChallengeStarterConfig(starterPresetId: string): ArchitectureConfig {
  const found = PRESET_BLUEPRINTS.find((p) => p.id === starterPresetId) ?? PRESET_BLUEPRINTS[0];
  return {
    ...found.config,
    dims: { ...found.config.dims },
  };
}

/**
 * Generates an editorial architectural archetype title & summary for the Spec Card!
 */
export function classifyArchetype(
  config: ArchitectureConfig,
  physics: PhysicsReport
): { title: string; subtitle: string } {
  if (config.hybrid === 'hyb_xlstm') {
    return {
      title: 'The Zero-KV Infinite Streamer',
      subtitle: 'Attention-free matrix recurrence with flat O(1) context memory.',
    };
  }
  if (config.attention === 'attn_csa_hca' && config.norm === 'norm_mhc') {
    return {
      title: 'The Hyper-Compressed Frontier Titan',
      subtitle: 'CSA/HCA extreme latent compression over 4-lane mHC residual rails.',
    };
  }
  if (config.ffn === 'ffn_ngram_moe') {
    return {
      title: 'The Engram Trillion Titan',
      subtitle: 'Shortcut MoE fused with massive O(1) N-Gram / Engram lookup memory tables.',
    };
  }
  if (config.attention === 'attn_difftpa' && (config.hybrid === 'hyb_mamba3' || config.hybrid === 'hyb_mamba2')) {
    return {
      title: 'The Differential State-Space Hybrid',
      subtitle: 'Mamba-3 SSM O(1) backbone paired with rank-4 noise-cancelling DiffTPA.',
    };
  }
  if (config.hybrid === 'hyb_mamba2' || config.hybrid === 'hyb_mamba3') {
    return {
      title: 'The State-Space Hybrid Cruiser',
      subtitle: 'Mostly Mamba-2 SSM backbone punctuated by surgical attention recall.',
    };
  }
  if (config.hybrid === 'hyb_deltanet' || config.hybrid === 'hyb_kda') {
    return {
      title: 'The Linear-Hybrid Assassin',
      subtitle: 'Gated delta-rule linear recurrence paired with periodic full attention.',
    };
  }
  if (config.ffn === 'ffn_looped_moe') {
    const lb = physics.loopBreakdown;
    return {
      title: 'The Sandwich Loop Titan',
      subtitle: lb
        ? `${lb.opening}+${lb.looped}×${lb.passes}+${lb.closing} → ${physics.effectiveDepth} effective layers with gated β_k progressive looping.`
        : `${physics.effectiveDepth} effective reasoning layers via looped sparse MoE depth sharing.`,
    };
  }
  if (config.ffn === 'ffn_looped_depth') {
    return {
      title: 'The Recurrent-Depth Thinker',
      subtitle: `${physics.effectiveDepth} effective reasoning layers compressed into ${config.dims.nLayers} physical blocks.`,
    };
  }
  if (
    (config.attention === 'attn_mla' || config.attention === 'attn_mla_dsa') &&
    physics.params.activeRatioPct <= 12
  ) {
    return {
      title: 'The Latent Sparse Architect',
      subtitle: 'Low-rank MLA KV compression fused with ultra-sparse MoE capacity.',
    };
  }
  if (config.hybrid === 'hyb_swa' && physics.vocabSize >= 240000) {
    return {
      title: 'The Polyglot Windowed Specialist',
      subtitle: 'Ultra-wide multilingual vocabulary over interleaved sliding-window attention.',
    };
  }
  if (config.attention === 'attn_mha' && config.position === 'pos_abs_learned') {
    return {
      title: 'The Classical Transformer Purist',
      subtitle: 'Pure 2019-era dense multi-head attention with learned positional embeddings.',
    };
  }
  if (physics.params.activeRatioPct < 25) {
    return {
      title: 'The Sparse Capacity Engine',
      subtitle: `Decouples ${(physics.params.total / 1e9).toFixed(0)}B knowledge from ${physics.params.activeRatioPct.toFixed(1)}% active compute.`,
    };
  }
  return {
    title: 'The Modern Dense Workhorse',
    subtitle: 'Balanced dense decoder engineered for predictable throughput & clean recall.',
  };
}
