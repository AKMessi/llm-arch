import type { ArchitectureConfig, GalleryModel, PresetBlueprint } from '../types/architecture';

export const PRESET_BLUEPRINTS: PresetBlueprint[] = [
  {
    id: 'llama-3-8b',
    name: 'Llama 3 8B',
    company: 'Meta',
    year: '2024',
    tagline: 'Modern Dense Transformer reference: GQA 4:1, RoPE, Pre-RMSNorm, SwiGLU, 128k tiktoken.',
    scaleBadge: '8.0B Dense',
    kvBadge: '128 KiB/tok',
    galleryModelName: 'Llama 3 8B',
    config: {
      name: 'Llama 3 8B Blueprint',
      tokenizer: 'tok_llama3_128k',
      position: 'pos_rope',
      norm: 'norm_pre_rms',
      attention: 'attn_gqa',
      hybrid: 'hyb_none',
      ffn: 'ffn_swiglu',
      output: 'out_untied',
      dims: {
        dModel: 4096,
        nLayers: 32,
        nHeads: 32,
        nKvHeads: 8,
        dFf: 14336,
        nExperts: 1,
        activeExperts: 1,
        nSharedExperts: 0,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 131072,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
  {
    id: 'deepseek-v3',
    name: 'DeepSeek V3 / R1',
    company: 'DeepSeek',
    year: '2024',
    tagline: 'Flagship Sparse MoE with MLA latent KV compression, shared expert & MTP-1.',
    scaleBadge: '671B · 37B Act',
    kvBadge: '68.6 KiB/tok',
    galleryModelName: 'DeepSeek V3',
    config: {
      name: 'DeepSeek V3 / R1 Blueprint',
      tokenizer: 'tok_deepseek_129k',
      position: 'pos_yarn_rope',
      norm: 'norm_pre_rms',
      attention: 'attn_mla',
      hybrid: 'hyb_none',
      ffn: 'ffn_moe_shared',
      output: 'out_mtp',
      dims: {
        dModel: 7168,
        nLayers: 61,
        nHeads: 128,
        nKvHeads: 128,
        dFf: 2048,
        nExperts: 256,
        activeExperts: 8,
        nSharedExperts: 1,
        densePrefixLayers: 3,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 131072,
        batchSize: 1,
        precision: 'fp8',
      },
    },
  },
  {
    id: 'qwen3-5-397b',
    name: 'Qwen3.5 397B-A17B',
    company: 'Alibaba Cloud',
    year: '2026',
    tagline: '3:1 Gated DeltaNet linear recurrence + Gated GQA hybrid with fine-grained MoE & 248k BPE.',
    scaleBadge: '397B · 17B Act',
    kvBadge: '30 KiB/tok',
    galleryModelName: 'Qwen3.5 397B',
    config: {
      name: 'Qwen3.5 Hybrid MoE Blueprint',
      tokenizer: 'tok_qwen_248k',
      position: 'pos_yarn_rope',
      norm: 'norm_rms_qknorm',
      attention: 'attn_gated_gqa',
      hybrid: 'hyb_deltanet',
      ffn: 'ffn_moe_shared',
      output: 'out_mtp',
      dims: {
        dModel: 4096,
        nLayers: 60,
        nHeads: 32,
        nKvHeads: 4,
        dFf: 2048,
        nExperts: 256,
        activeExperts: 8,
        nSharedExperts: 1,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 262144,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
  {
    id: 'gemma-4-26b-a4b',
    name: 'Gemma 4 26B-A4B',
    company: 'Google',
    year: '2026',
    tagline: '5:1 Sliding-Window / Global GQA with p-RoPE, Peri-Norm 4×, Sparse MoE & 262k vocab.',
    scaleBadge: '25.2B · 3.8B Act',
    kvBadge: 'SWA-Capped',
    galleryModelName: 'Gemma 4 26B-A4B',
    config: {
      name: 'Gemma 4 26B-A4B Blueprint',
      tokenizer: 'tok_gemma_262k',
      position: 'pos_partial_rope',
      norm: 'norm_peri_gemma',
      attention: 'attn_gqa',
      hybrid: 'hyb_swa',
      ffn: 'ffn_moe_sparse',
      output: 'out_softcap',
      dims: {
        dModel: 2816,
        nLayers: 30,
        nHeads: 16,
        nKvHeads: 4,
        dFf: 1024,
        nExperts: 128,
        activeExperts: 8,
        nSharedExperts: 1,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 1024,
        hybridRatio: 5,
        loopPasses: 1,
        contextLength: 262144,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
  {
    id: 'deepseek-v4-pro',
    name: 'DeepSeek V4-Pro',
    company: 'DeepSeek',
    year: '2026',
    tagline: 'Trillion-scale 1M-ctx titan: CSA/HCA ultra-compressed MLA + mHC multi-stream residuals.',
    scaleBadge: '1.6T · 49B Act',
    kvBadge: '7.7 KiB/tok',
    galleryModelName: 'DeepSeek V4-Pro',
    config: {
      name: 'DeepSeek V4-Pro Blueprint',
      tokenizer: 'tok_deepseek_129k',
      position: 'pos_yarn_rope',
      norm: 'norm_mhc',
      attention: 'attn_csa_hca',
      hybrid: 'hyb_none',
      ffn: 'ffn_moe_shared',
      output: 'out_mtp',
      dims: {
        dModel: 8192,
        nLayers: 61,
        nHeads: 128,
        nKvHeads: 128,
        dFf: 2560,
        nExperts: 384,
        activeExperts: 8,
        nSharedExperts: 2,
        densePrefixLayers: 3,
        mlaKvLoraRank: 256,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 1048576,
        batchSize: 1,
        precision: 'fp8',
      },
    },
  },
  {
    id: 'nemotron-3-super',
    name: 'Nemotron 3 Super',
    company: 'NVIDIA',
    year: '2026',
    tagline: 'Mostly Mamba-2 SSM with only 8 GQA layers + Latent MoE — 8 KiB/tok KV at 1M context!',
    scaleBadge: '120B · 12B Act',
    kvBadge: '8 KiB/tok',
    galleryModelName: 'Nemotron 3.5 Super VL 120B-A12B',
    config: {
      name: 'Nemotron 3 Super Hybrid Blueprint',
      tokenizer: 'tok_llama3_128k',
      position: 'pos_pure_nope',
      norm: 'norm_pre_rms',
      attention: 'attn_gqa',
      hybrid: 'hyb_mamba2',
      ffn: 'ffn_latent_moe',
      output: 'out_mtp',
      dims: {
        dModel: 4096,
        nLayers: 48,
        nHeads: 32,
        nKvHeads: 4,
        dFf: 2048,
        nExperts: 128,
        activeExperts: 6,
        nSharedExperts: 1,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 5,
        loopPasses: 1,
        contextLength: 1048576,
        batchSize: 1,
        precision: 'fp8',
      },
    },
  },
  {
    id: 'qwen3-32b',
    name: 'Qwen3 32B Dense',
    company: 'Alibaba Cloud',
    year: '2025',
    tagline: 'Clean dense powerhouse: GQA + QK-Norm + 152k Byte-BPE + SwiGLU across 64 layers.',
    scaleBadge: '32B Dense',
    kvBadge: '256 KiB/tok',
    galleryModelName: 'Qwen3 32B',
    config: {
      name: 'Qwen3 32B Blueprint',
      tokenizer: 'tok_qwen_152k',
      position: 'pos_rope',
      norm: 'norm_rms_qknorm',
      attention: 'attn_gqa',
      hybrid: 'hyb_none',
      ffn: 'ffn_swiglu',
      output: 'out_untied',
      dims: {
        dModel: 5120,
        nLayers: 64,
        nHeads: 40,
        nKvHeads: 8,
        dFf: 25600,
        nExperts: 1,
        activeExperts: 1,
        nSharedExperts: 0,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 131072,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
  {
    id: 'olmo-3-32b',
    name: 'OLMo 3 32B',
    company: 'Ai2',
    year: '2025',
    tagline: 'Open-science dense stack: Post-RMSNorm + QK-Norm + 3:1 SWA/Global GQA.',
    scaleBadge: '32B Dense',
    kvBadge: '256 KiB/tok',
    galleryModelName: 'OLMo 3 32B',
    config: {
      name: 'OLMo 3 32B Blueprint',
      tokenizer: 'tok_llama3_128k',
      position: 'pos_rope',
      norm: 'norm_post_olmo',
      attention: 'attn_gqa',
      hybrid: 'hyb_swa',
      ffn: 'ffn_swiglu',
      output: 'out_untied',
      dims: {
        dModel: 5120,
        nLayers: 64,
        nHeads: 40,
        nKvHeads: 8,
        dFf: 27648,
        nExperts: 1,
        activeExperts: 1,
        nSharedExperts: 0,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 65536,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
  {
    id: 'mistral-small-4',
    name: 'Mistral Small 4',
    company: 'Mistral AI',
    year: '2026',
    tagline: 'Mistral shifts to MLA + Sparse MoE + Tekken 131k for ultra-lean 256k context.',
    scaleBadge: '119B · 6.6B Act',
    kvBadge: '22.5 KiB/tok',
    galleryModelName: 'Mistral Small 4',
    config: {
      name: 'Mistral Small 4 Blueprint',
      tokenizer: 'tok_tekken_131k',
      position: 'pos_yarn_rope',
      norm: 'norm_pre_rms',
      attention: 'attn_mla',
      hybrid: 'hyb_none',
      ffn: 'ffn_moe_sparse',
      output: 'out_untied',
      dims: {
        dModel: 4096,
        nLayers: 36,
        nHeads: 32,
        nKvHeads: 32,
        dFf: 1536,
        nExperts: 128,
        activeExperts: 4,
        nSharedExperts: 1,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 262144,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
  {
    id: 'gemma-4-e4b',
    name: 'Gemma 4 E4B Edge',
    company: 'Google',
    year: '2026',
    tagline: 'Edge miracle: Per-Layer Embeddings + Cross-Layer KV Sharing + 5:1 SWA + 262k vocab.',
    scaleBadge: '8B (4.5B Eff)',
    kvBadge: '35 KiB/tok',
    galleryModelName: 'Gemma 4 E4B',
    config: {
      name: 'Gemma 4 E4B Edge Blueprint',
      tokenizer: 'tok_gemma_262k',
      position: 'pos_ple',
      norm: 'norm_peri_gemma',
      attention: 'attn_kv_sharing',
      hybrid: 'hyb_swa',
      ffn: 'ffn_geglu',
      output: 'out_softcap',
      dims: {
        dModel: 2560,
        nLayers: 42,
        nHeads: 8,
        nKvHeads: 2,
        dFf: 10240,
        nExperts: 1,
        activeExperts: 1,
        nSharedExperts: 0,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 1024,
        hybridRatio: 5,
        loopPasses: 1,
        contextLength: 131072,
        batchSize: 1,
        precision: 'int4',
      },
    },
  },
  {
    id: 'gpt-2-xl',
    name: 'GPT-2 XL (2019)',
    company: 'OpenAI',
    year: '2019',
    tagline: 'Historical baseline: Vanilla MHA, Learned AbsPos, Pre-LayerNorm, GELU MLP, 50k BPE.',
    scaleBadge: '1.5B Dense',
    kvBadge: '300 KiB/tok',
    galleryModelName: 'GPT-2 XL 1.5B',
    config: {
      name: 'GPT-2 XL (2019 Baseline)',
      tokenizer: 'tok_gpt2_50k',
      position: 'pos_abs_learned',
      norm: 'norm_pre_ln',
      attention: 'attn_mha',
      hybrid: 'hyb_none',
      ffn: 'ffn_gelu',
      output: 'out_tied',
      dims: {
        dModel: 1600,
        nLayers: 48,
        nHeads: 25,
        nKvHeads: 25,
        dFf: 6400,
        nExperts: 1,
        activeExperts: 1,
        nSharedExperts: 0,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 1024,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 1024,
        batchSize: 1,
        precision: 'fp32',
      },
    },
  },
  {
    id: 'xlstm-7b',
    name: 'xLSTM 7B',
    company: 'NXAI',
    year: '2025',
    tagline: 'Attention-free mLSTM matrix recurrence with 0 B/token KV-cache & constant O(1) memory.',
    scaleBadge: '7.0B Recurrent',
    kvBadge: '0 B/tok!',
    galleryModelName: 'xLSTM 7B',
    config: {
      name: 'xLSTM 7B Recurrent Blueprint',
      tokenizer: 'tok_gpt2_50k',
      position: 'pos_pure_nope',
      norm: 'norm_pre_rms',
      attention: 'attn_gqa',
      hybrid: 'hyb_xlstm',
      ffn: 'ffn_swiglu',
      output: 'out_untied',
      dims: {
        dModel: 4096,
        nLayers: 32,
        nHeads: 8,
        nKvHeads: 8,
        dFf: 10880,
        nExperts: 1,
        activeExperts: 1,
        nSharedExperts: 0,
        densePrefixLayers: 0,
        mlaKvLoraRank: 512,
        mlaQkRopeDim: 64,
        swaWindow: 4096,
        hybridRatio: 3,
        loopPasses: 1,
        contextLength: 262144,
        batchSize: 1,
        precision: 'bf16',
      },
    },
  },
];

/**
 * Intelligently maps ANY of the 120 real models from Sebastian Raschka's
 * LLM Architecture Gallery into a faithful playable ArchitectureConfig!
 */
export function galleryModelToConfig(model: GalleryModel): ArchitectureConfig {
  // Check exact preset match first
  const presetMatch = PRESET_BLUEPRINTS.find(
    (p) => p.galleryModelName.toLowerCase() === model.name.toLowerCase()
  );
  if (presetMatch) {
    return {
      ...presetMatch.config,
      dims: { ...presetMatch.config.dims },
    };
  }

  const c = new Set(model.concepts);
  const attnText = (model.attention + ' ' + model.layerMix + ' ' + model.highlight).toLowerCase();
  const isMoE = c.has('moe') || /moe/i.test(model.decoderType);

  // 1. Tokenizer selection by vocab size or company
  let tokenizer = 'tok_llama3_128k';
  const v = model.vocabSize ?? 0;
  if (v >= 255000 || /gemma|cohere|command|zaya/i.test(model.name)) tokenizer = 'tok_gemma_262k';
  else if (v >= 220000 || /qwen3\.[568]|btl-3|ornith/i.test(model.name)) tokenizer = 'tok_qwen_248k';
  else if (v >= 145000 || /qwen|mimo|reflection|glm/i.test(model.name)) tokenizer = 'tok_qwen_152k';
  else if (/mistral/i.test(model.name) || (v >= 130000 && v <= 132000)) tokenizer = 'tok_tekken_131k';
  else if (/deepseek/i.test(model.name)) tokenizer = 'tok_deepseek_129k';
  else if (v > 0 && v <= 55000) tokenizer = 'tok_gpt2_50k';

  // 2. Position encoding
  let position = 'pos_rope';
  if (/learned absolute/i.test(attnText)) position = 'pos_abs_learned';
  else if (c.has('per_layer_embeddings')) position = 'pos_ple';
  else if (c.has('nope') && c.has('rope')) {
    position = /p-rope|partial rope/i.test(attnText)
      ? 'pos_partial_rope'
      : 'pos_local_rope_global_nope';
  } else if (c.has('nope') && !c.has('rope')) {
    position = 'pos_pure_nope';
  } else if (/p-rope|partial rope/i.test(attnText)) {
    position = 'pos_partial_rope';
  } else if ((model.contextTokens ?? 0) >= 260000) {
    position = 'pos_yarn_rope';
  }

  // 3. Normalization & Residuals
  let norm = 'norm_pre_rms';
  if (/gpt-2/i.test(model.name)) norm = 'norm_pre_ln';
  else if (c.has('mhc')) norm = 'norm_mhc';
  else if (c.has('attention_residuals') || c.has('gated_residuals')) norm = 'norm_attn_res';
  else if (/gemma/i.test(model.name)) norm = 'norm_peri_gemma';
  else if (/olmo/i.test(model.name)) norm = 'norm_post_olmo';
  else if (c.has('qk_norm')) norm = 'norm_rms_qknorm';

  // 4. Primary Attention
  let attention = 'attn_gqa';
  if (c.has('csa_hca')) attention = 'attn_csa_hca';
  else if (c.has('mla') && c.has('deepseek_sparse_attention')) attention = 'attn_mla_dsa';
  else if (c.has('mla')) attention = 'attn_mla';
  else if (c.has('kv_sharing')) attention = 'attn_kv_sharing';
  else if (c.has('gated_attention')) attention = 'attn_gated_gqa';
  else if (c.has('mha') || /\bmha\b/i.test(attnText)) attention = 'attn_mha';
  else if (/multi-query/i.test(attnText)) attention = 'attn_mqa';

  // 5. Hybrid Layer Interleave
  let hybrid = 'hyb_none';
  let hybridRatio = 3;
  if (/mlstm|xlstm/i.test(attnText) || /recurrent/i.test(model.decoderType)) {
    hybrid = 'hyb_xlstm';
  } else if (/mamba-2|mamba/i.test(attnText)) {
    hybrid = 'hyb_mamba2';
    hybridRatio = 5;
  } else if (/kimi delta|kda/i.test(attnText)) {
    hybrid = 'hyb_kda';
    const rMatch = attnText.match(/(\d+):1/);
    if (rMatch) hybridRatio = Math.min(7, Math.max(2, parseInt(rMatch[1], 10)));
  } else if (c.has('gated_deltanet') || /deltanet/i.test(attnText)) {
    hybrid = 'hyb_deltanet';
    const rMatch = attnText.match(/(\d+):1/);
    if (rMatch) hybridRatio = Math.min(7, Math.max(2, parseInt(rMatch[1], 10)));
  } else if (/liv convolution|shortconv plus/i.test(attnText)) {
    hybrid = 'hyb_shortconv';
    hybridRatio = 3;
  } else if (c.has('swa') || /sliding-window/i.test(attnText)) {
    hybrid = 'hyb_swa';
    const rMatch = attnText.match(/(\d+):1/);
    if (rMatch) hybridRatio = Math.min(6, Math.max(1, parseInt(rMatch[1], 10)));
    else if (/alternating/i.test(attnText)) hybridRatio = 1;
  }

  // 6. FFN / MoE
  let ffn = 'ffn_swiglu';
  let loopPasses = 1;
  if (c.has('looped_depth_sharing')) {
    ffn = 'ffn_looped_depth';
    const passMatch = model.layerMix.match(/×\s*(\d+)\s*pass/i);
    loopPasses = passMatch ? Math.min(4, Math.max(2, parseInt(passMatch[1], 10))) : 2;
  } else if (c.has('latent_moe')) {
    ffn = 'ffn_latent_moe';
  } else if (isMoE) {
    const activeRatio =
      model.totalParamsB && model.activeParamsB ? model.activeParamsB / model.totalParamsB : 0.1;
    ffn = activeRatio < 0.08 || /deepseek|qwen3\.5|glm-5|kimi/i.test(model.name)
      ? 'ffn_moe_shared'
      : 'ffn_moe_sparse';
  } else if (/gemma/i.test(model.name)) {
    ffn = 'ffn_geglu';
  } else if (/gpt-2/i.test(model.name)) {
    ffn = 'ffn_gelu';
  }

  // 7. Output Head & Objectives
  let output = 'out_untied';
  if (/attention sinks/i.test(attnText)) output = 'out_sinks_mtp';
  else if (c.has('mtp')) output = 'out_mtp';
  else if (/gemma 3/i.test(model.name)) output = 'out_softcap';
  else if (/tied/i.test(model.highlight) || (model.totalParamsB ?? 10) <= 4) output = 'out_tied';

  // Estimate layer count from layerMix string (sum of numbers before layer types)
  let nLayers = 32;
  const nums = model.layerMix.match(/\b(\d+)\b/g);
  if (nums && nums.length > 0) {
    const parsed = nums.map((x) => parseInt(x, 10)).filter((x) => x > 0 && x <= 140);
    if (parsed.length === 1) nLayers = parsed[0];
    else if (parsed.length >= 2) {
      // Handle e.g. "45 sliding-window + 15 global" or "6 GQA + 23 Mamba-2 + 23 MoE"
      if (/moe|ffn/i.test(model.layerMix) && parsed.length >= 3) {
        nLayers = parsed[0] + parsed[1];
      } else {
        nLayers = parsed[0] + parsed[1];
      }
    }
  }
  nLayers = Math.max(12, Math.min(96, nLayers));

  // Scale dModel & MoE dimensions realistically to totalParamsB
  const totalB = model.totalParamsB ?? 8;
  const activeB = model.activeParamsB ?? totalB;
  let dModel = 4096;
  let nHeads = 32;
  let nKvHeads = 8;
  let dFf = 14336;
  let nExperts = 1;
  let activeExperts = 1;
  let nSharedExperts = 0;

  if (isMoE) {
    if (totalB >= 600) {
      dModel = 7168;
      nHeads = 64;
      nKvHeads = 8;
      nExperts = 256;
      activeExperts = 8;
      nSharedExperts = 1;
      dFf = 2048;
    } else if (totalB >= 200) {
      dModel = 5120;
      nHeads = 40;
      nKvHeads = 8;
      nExperts = 128;
      activeExperts = 8;
      nSharedExperts = 1;
      dFf = 1536;
    } else if (totalB >= 70) {
      dModel = 4096;
      nHeads = 32;
      nKvHeads = 8;
      nExperts = 128;
      activeExperts = 6;
      nSharedExperts = 1;
      dFf = 1536;
    } else {
      dModel = 2560;
      nHeads = 32;
      nKvHeads = 4;
      nExperts = 64;
      activeExperts = 6;
      nSharedExperts = 1;
      dFf = 1024;
    }
    if (activeB && totalB > 0) {
      const targetRatio = Math.max(0.03, Math.min(0.25, activeB / totalB));
      activeExperts = Math.max(2, Math.min(16, Math.round(nExperts * targetRatio)));
    }
  } else {
    if (totalB >= 60) {
      dModel = 8192;
      nHeads = 64;
      nKvHeads = 8;
      dFf = 28672;
    } else if (totalB >= 22) {
      dModel = 5120;
      nHeads = 40;
      nKvHeads = 8;
      dFf = 20480;
    } else if (totalB >= 6) {
      dModel = 4096;
      nHeads = 32;
      nKvHeads = 8;
      dFf = 14336;
    } else if (totalB >= 2) {
      dModel = 2560;
      nHeads = 20;
      nKvHeads = 4;
      dFf = 8192;
    } else {
      dModel = 1536;
      nHeads = 12;
      nKvHeads = 4;
      dFf = 4096;
    }
  }

  const ctx = model.contextTokens && model.contextTokens >= 1024
    ? Math.min(1048576, model.contextTokens)
    : 131072;

  return {
    name: `${model.name} (Gallery Clone)`,
    tokenizer,
    position,
    norm,
    attention,
    hybrid,
    ffn,
    output,
    dims: {
      dModel,
      nLayers,
      nHeads,
      nKvHeads,
      dFf,
      nExperts,
      activeExperts,
      nSharedExperts,
      densePrefixLayers: ffn === 'ffn_moe_shared' ? 3 : 0,
      mlaKvLoraRank: attention === 'attn_csa_hca' ? 256 : 512,
      mlaQkRopeDim: 64,
      swaWindow: 4096,
      hybridRatio,
      loopPasses,
      contextLength: ctx,
      batchSize: 1,
      precision: totalB >= 200 ? 'fp8' : 'bf16',
    },
  };
}
