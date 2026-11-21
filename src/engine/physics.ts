import { COMPONENT_MAP } from '../data/components';
import type {
  ArchitectureConfig,
  HardwareFit,
  KvCurvePoint,
  LayerSpec,
  PhysicsReport,
  PrecisionMode,
} from '../types/architecture';

export function precisionBytes(p: PrecisionMode): number {
  switch (p) {
    case 'fp32':
      return 4;
    case 'bf16':
      return 2;
    case 'fp8':
      return 1;
    case 'int4':
      return 0.5;
  }
}

export function kvPrecisionBytes(p: PrecisionMode): number {
  // Even with INT4 weights, KV-cache is commonly stored in FP8 or BF16; reflect realistic FP8 for int4/fp8
  if (p === 'fp32') return 4;
  if (p === 'bf16') return 2;
  return 1; // FP8 KV cache for fp8 & int4 deployment
}

export function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 100e9 ? 1 : 2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return `${(n / 1e3).toFixed(0)}K`;
}

export function formatKiB(kib: number): string {
  if (kib === 0) return '0 B';
  if (kib < 1) return `${Math.round(kib * 1024)} B`;
  if (kib >= 1024) return `${(kib / 1024).toFixed(2)} MiB`;
  return `${kib >= 100 ? Math.round(kib) : kib.toFixed(1)} KiB`;
}

export function formatGb(gb: number): string {
  if (gb < 0.01) return `${(gb * 1024).toFixed(1)} MB`;
  if (gb < 1) return `${(gb * 1024).toFixed(0)} MB`;
  if (gb >= 1000) return `${(gb / 1024).toFixed(2)} TB`;
  return `${gb >= 100 ? gb.toFixed(0) : gb.toFixed(1)} GB`;
}

export function formatContext(ctx: number): string {
  if (ctx >= 1048576) return `${(ctx / 1048576).toFixed(ctx % 1048576 === 0 ? 0 : 1)}M`;
  if (ctx >= 1024) return `${Math.round(ctx / 1024)}K`;
  return `${ctx}`;
}

export function computePhysics(config: ArchitectureConfig): PhysicsReport {
  const { dims } = config;
  const tokComp = COMPONENT_MAP[config.tokenizer];
  const posComp = COMPONENT_MAP[config.position];
  const normComp = COMPONENT_MAP[config.norm];
  const attnComp = COMPONENT_MAP[config.attention];
  const hybComp = COMPONENT_MAP[config.hybrid];
  const ffnComp = COMPONENT_MAP[config.ffn];
  const outComp = COMPONENT_MAP[config.output];

  const V = tokComp?.vocabSize ?? 128256;
  const d = dims.dModel;
  const L = dims.nLayers;
  const H = Math.max(1, dims.nHeads);
  const dHead = Math.max(16, Math.floor(d / H));
  const nKv = Math.max(1, Math.min(H, dims.nKvHeads));
  const loopPasses = config.ffn === 'ffn_looped_depth' ? Math.max(2, dims.loopPasses) : 1;
  const effectiveDepth = L * loopPasses;

  const isMoE =
    config.ffn === 'ffn_moe_sparse' ||
    config.ffn === 'ffn_moe_shared' ||
    config.ffn === 'ffn_latent_moe';
  const isLatentMoE = config.ffn === 'ffn_latent_moe';
  const isPureRecurrent = config.hybrid === 'hyb_xlstm';
  const isKvShared = config.attention === 'attn_kv_sharing';

  // Primary attention KV bytes/token/layer in BF16 (2 bytes per scalar)
  const primaryKvPerLayerBf16 = (() => {
    switch (config.attention) {
      case 'attn_mha':
        return 2 * H * dHead * 2;
      case 'attn_mqa':
        return 2 * 1 * dHead * 2;
      case 'attn_gqa':
      case 'attn_gated_gqa':
      case 'attn_kv_sharing':
        return 2 * nKv * dHead * 2;
      case 'attn_mla':
      case 'attn_mla_dsa':
        return (dims.mlaKvLoraRank + dims.mlaQkRopeDim) * 2;
      case 'attn_csa_hca':
        // DeepSeek V4 / ERNIE 5.5 ultra-compressed hierarchical latent (~64-128 dim effective)
        return Math.max(64, Math.round((dims.mlaKvLoraRank + dims.mlaQkRopeDim) * 0.22)) * 2;
      default:
        return 2 * nKv * dHead * 2;
    }
  })();

  // Build exact Layer-by-Layer schedule (L physical layers)
  const layers: LayerSpec[] = [];
  const ratio = Math.max(1, dims.hybridRatio);
  const densePrefix = isMoE ? Math.min(L - 1, Math.max(0, dims.densePrefixLayers)) : 0;

  let primaryCount = 0;
  let secondaryCount = 0;
  let swaCount = 0;
  let kvFullBytesPerTokenBf16 = 0;
  let kvSwaBytesPerTokenBf16 = 0;
  let recurrentLayers = 0;

  for (let i = 0; i < L; i++) {
    const idx = i + 1;
    let mixerKind: LayerSpec['mixerKind'] = 'primary';
    let mixerLabel = attnComp?.shortName ?? 'GQA';
    let layerKvBf16 = primaryKvPerLayerBf16;

    if (isPureRecurrent) {
      mixerKind = 'mlstm';
      mixerLabel = 'mLSTM';
      layerKvBf16 = 0;
      secondaryCount++;
      recurrentLayers++;
    } else if (config.hybrid !== 'hyb_none') {
      // Every (ratio + 1)th layer is Primary Global; the rest are Secondary
      const isGlobalSlot = idx % (ratio + 1) === 0 || idx === L;
      if (!isGlobalSlot) {
        secondaryCount++;
        if (config.hybrid === 'hyb_swa') {
          mixerKind = 'swa';
          mixerLabel = `SWA (${formatContext(dims.swaWindow)})`;
          swaCount++;
          // SWA stores KV up to window W
          layerKvBf16 = primaryKvPerLayerBf16;
        } else if (config.hybrid === 'hyb_deltanet') {
          mixerKind = 'deltanet';
          mixerLabel = 'Gated DeltaNet';
          layerKvBf16 = 0;
          recurrentLayers++;
        } else if (config.hybrid === 'hyb_kda') {
          mixerKind = 'kda';
          mixerLabel = 'Kimi KDA';
          layerKvBf16 = 0;
          recurrentLayers++;
        } else if (config.hybrid === 'hyb_mamba2') {
          mixerKind = 'mamba2';
          mixerLabel = 'Mamba-2 SSM';
          layerKvBf16 = 0;
          recurrentLayers++;
        } else if (config.hybrid === 'hyb_shortconv') {
          mixerKind = 'shortconv';
          mixerLabel = 'LIV ShortConv';
          layerKvBf16 = 0;
          recurrentLayers++;
        }
      } else {
        primaryCount++;
      }
    } else {
      primaryCount++;
    }

    // Cross-layer KV sharing: upper 45% of attention layers reuse lower layer KV caches
    let kvSharedSource: number | undefined;
    if (isKvShared && layerKvBf16 > 0 && idx > Math.ceil(L * 0.55)) {
      kvSharedSource = Math.max(1, idx - Math.floor(L * 0.45));
      layerKvBf16 = 0;
    }

    if (mixerKind === 'swa') {
      kvSwaBytesPerTokenBf16 += layerKvBf16;
    } else {
      kvFullBytesPerTokenBf16 += layerKvBf16;
    }

    const ffnKind: LayerSpec['ffnKind'] =
      !isMoE || idx <= densePrefix ? 'dense' : isLatentMoE ? 'latent_moe' : 'moe';

    layers.push({
      index: idx,
      mixerKind,
      mixerLabel,
      ffnKind,
      kvBytesPerTokenBf16: layerKvBf16 * loopPasses,
      kvSharedSource,
    });
  }

  // Account for looped depth multiplying KV cache per pass (like Ouro-Thinking & Dory)
  const kvNominalBytesPerTokenBf16 = (kvFullBytesPerTokenBf16 + kvSwaBytesPerTokenBf16) * loopPasses;
  const kvKiBPerTokenBf16 = kvNominalBytesPerTokenBf16 / 1024;
  const kvScaleRatio = kvPrecisionBytes(dims.precision) / 2;
  const kvKiBPerTokenSelected = kvKiBPerTokenBf16 * kvScaleRatio;

  // Exact KV cache at target context length (respecting SWA window cap!)
  const calcKvGbAtContext = (ctx: number, bytesRatio: number): number => {
    const fullBytes = kvFullBytesPerTokenBf16 * loopPasses * ctx * bytesRatio * dims.batchSize;
    const swaBytes =
      kvSwaBytesPerTokenBf16 *
      loopPasses *
      Math.min(ctx, dims.swaWindow) *
      bytesRatio *
      dims.batchSize;
    return (fullBytes + swaBytes) / (1024 ** 3);
  };

  const kvCacheGbAtContext = calcKvGbAtContext(dims.contextLength, kvScaleRatio);

  // Constant recurrent state for DeltaNet / KDA / Mamba-2 / mLSTM layers (fixed regardless of T!)
  const recurrentStateBytes =
    recurrentLayers * H * dHead * dHead * 2 * dims.batchSize; // matrix state [H, d_h, d_h] in BF16
  const recurrentStateMb = recurrentStateBytes / (1024 ** 2);

  // ==================== EXACT PARAMETER CALCULATIONS ====================
  const inputEmbedParams = V * d;
  const absPosParams = config.position === 'pos_abs_learned' ? dims.contextLength * d : 0;
  // Per-Layer Embeddings (Gemma 4 E-series style: V × 256 × L lightweight table)
  const pleParams = config.position === 'pos_ple' ? V * 256 * L : 0;
  const embeddings = inputEmbedParams + absPosParams + pleParams;

  // Per-layer Attention / Mixer parameters
  let attentionTotal = 0;
  for (const layer of layers) {
    if (layer.mixerKind === 'deltanet' || layer.mixerKind === 'kda') {
      // Q, K, V, Gate, Beta/Alpha + ShortConv + Out proj ≈ 4.5 d^2
      attentionTotal += 4.5 * d * d;
    } else if (layer.mixerKind === 'mamba2') {
      // Expand 2× in_proj, conv, dt/B/C, out_proj ≈ 6 d^2
      attentionTotal += 6 * d * d;
    } else if (layer.mixerKind === 'mlstm') {
      attentionTotal += 5.5 * d * d;
    } else if (layer.mixerKind === 'shortconv') {
      attentionTotal += 2.2 * d * d;
    } else {
      // Transformer attention
      if (config.attention === 'attn_mha') {
        attentionTotal += 4 * d * d;
      } else if (config.attention === 'attn_mqa') {
        attentionTotal += 2 * d * d + 2 * d * dHead;
      } else if (config.attention === 'attn_gqa' || config.attention === 'attn_kv_sharing') {
        attentionTotal += 2 * d * d + 2 * d * (nKv * dHead);
      } else if (config.attention === 'attn_gated_gqa') {
        // Extra output sigmoid gate W_g (d × d)
        attentionTotal += 3 * d * d + 2 * d * (nKv * dHead);
      } else if (config.attention === 'attn_mla' || config.attention === 'attn_mla_dsa' || config.attention === 'attn_csa_hca') {
        const dc = dims.mlaKvLoraRank;
        const dr = dims.mlaQkRopeDim;
        const dq = Math.min(d, 1536);
        const qProj = d * dq + dq * H * (dHead + dr);
        const kvProj = d * (dc + dr) + dc * H * (2 * dHead);
        const oProj = H * dHead * d;
        const sparseIdx = config.attention !== 'attn_mla' ? 0.15 * d * d : 0;
        attentionTotal += qProj + kvProj + oProj + sparseIdx;
      }
    }
  }

  // Per-layer FFN / MoE parameters
  let ffnTotal = 0;
  let ffnActive = 0;
  const isTwoMatrixMlp = config.ffn === 'ffn_gelu';
  const ffnMatrixFactor = isTwoMatrixMlp ? 2 : 3; // SwiGLU/GeGLU have W_gate, W_up, W_down

  for (const layer of layers) {
    if (layer.ffnKind === 'dense') {
      const denseMid = isMoE ? Math.max(dims.dFf * 4, Math.round(d * 2.7)) : dims.dFf;
      const p = ffnMatrixFactor * d * denseMid;
      ffnTotal += p;
      ffnActive += p;
    } else if (layer.ffnKind === 'latent_moe') {
      const dLat = Math.max(512, Math.round(d * 0.5));
      const latProj = 2 * d * dLat;
      const expertParams = 3 * dLat * dims.dFf;
      const router = d * dims.nExperts;
      const shared = dims.nSharedExperts * 3 * d * dims.dFf;
      ffnTotal += latProj + router + shared + dims.nExperts * expertParams;
      ffnActive += latProj + router + shared + Math.min(dims.nExperts, dims.activeExperts) * expertParams;
    } else {
      // Standard or Fine-Grained + Shared MoE
      const expertParams = 3 * d * dims.dFf;
      const router = d * dims.nExperts;
      const shared =
        config.ffn === 'ffn_moe_shared' ? dims.nSharedExperts * 3 * d * dims.dFf : 0;
      ffnTotal += router + shared + dims.nExperts * expertParams;
      ffnActive += router + shared + Math.min(dims.nExperts, dims.activeExperts) * expertParams;
    }
  }

  // Norms & Residual topology parameters
  const normsPerLayer =
    config.norm === 'norm_peri_gemma'
      ? 4
      : config.norm === 'norm_pre_ln'
        ? 4 // γ and β × 2
        : 2;
  const qkNormParams =
    config.norm === 'norm_rms_qknorm' ||
    config.norm === 'norm_peri_gemma' ||
    config.norm === 'norm_post_olmo'
      ? 2 * dHead * L
      : 0;
  const mhcParams = config.norm === 'norm_mhc' ? L * (d * 16 + 64) : 0;
  const attnResParams = config.norm === 'norm_attn_res' ? L * d * 4 : 0;
  const normsAndMisc = L * normsPerLayer * d + d + qkNormParams + mhcParams + attnResParams;

  // Output LM Head + MTP auxiliary layers
  const isTied = config.output === 'out_tied' || config.output === 'out_softcap';
  const hasMtp = config.output === 'out_mtp' || config.output === 'out_sinks_mtp';
  const baseHead = isTied ? 0 : V * d;
  const mtpParams = hasMtp ? 2 * d * d + (ffnActive / Math.max(1, L)) : 0;
  const outputHead = baseHead + mtpParams;

  const total = embeddings + attentionTotal + ffnTotal + normsAndMisc + outputHead;
  // PLE is lookup-only per token (negligible active compute); active params = non-PLE embed + attn + ffnActive + norms + head
  const activeEmbed = inputEmbedParams + (config.position === 'pos_ple' ? 256 * L : 0);
  const active = activeEmbed + attentionTotal + ffnActive + normsAndMisc + outputHead;
  const activeRatioPct = total > 0 ? (active / total) * 100 : 100;

  // ==================== VRAM, FLOPS & ROOFLINE ====================
  const wBytes = precisionBytes(dims.precision);
  const weightsGb = (total * wBytes) / (1024 ** 3);
  const mhcActMul = config.norm === 'norm_mhc' ? 2.4 : 1;
  const activationsGb =
    ((dims.batchSize * d * effectiveDepth * 6 * 2 * mhcActMul) + recurrentStateBytes) /
    (1024 ** 3);
  const totalVramGb = weightsGb + kvCacheGbAtContext + activationsGb;

  // Active decode FLOPs/token: ~2 * active_non_embed * loopPasses + attention dot-product cost
  const activeNonEmbed = Math.max(1e6, active - activeEmbed);
  const matmulFlops = 2 * activeNonEmbed * loopPasses;
  const attnLayersCount = primaryCount + swaCount;
  const effAttnSpan =
    config.attention === 'attn_mla_dsa' || config.attention === 'attn_csa_hca'
      ? Math.min(dims.contextLength, 4096) // sparse top-k blocks
      : config.hybrid === 'hyb_swa'
        ? (primaryCount * dims.contextLength + swaCount * Math.min(dims.contextLength, dims.swaWindow)) /
          Math.max(1, attnLayersCount)
        : dims.contextLength;
  const attnDotFlops = 2 * attnLayersCount * loopPasses * effAttnSpan * d * 2;
  const flopsPerTokenActive = matmulFlops + attnDotFlops;
  const attentionFlopsSharePct = Math.min(
    95,
    (attnDotFlops / Math.max(1, flopsPerTokenActive)) * 100
  );

  // Memory read per decode token (active weights + KV cache read for 1 sequence)
  const activeWeightBytesRead = active * wBytes;
  const kvBytesReadPerToken = kvCacheGbAtContext * (1024 ** 3) / Math.max(1, dims.batchSize);
  const totalBytesRead = activeWeightBytesRead + kvBytesReadPerToken + recurrentStateBytes;
  const arithmeticIntensity = flopsPerTokenActive / Math.max(1, totalBytesRead);
  const bottleneck: PhysicsReport['bottleneck'] =
    arithmeticIntensity < 45
      ? 'Memory-Bandwidth Bound'
      : arithmeticIntensity > 180
        ? 'Compute Bound'
        : 'Balanced';

  // Estimated decode tok/s (H100 SXM5: 3.35 TB/s, 989 TFLOPs BF16; M4 Max: 546 GB/s)
  const mtpSpeedup = hasMtp ? 1.65 : 1.0;
  const h100MemTokSec = (3.35e12 * 0.72) / Math.max(1, totalBytesRead);
  const h100CompTokSec = (989e12 * 0.45) / Math.max(1, flopsPerTokenActive);
  const estDecodeTokSecH100 = Math.min(h100MemTokSec, h100CompTokSec) * mtpSpeedup;

  const m4MemTokSec = (546e9 * 0.7) / Math.max(1, totalBytesRead);
  const m4CompTokSec = (40e12 * 0.4) / Math.max(1, flopsPerTokenActive);
  const estDecodeTokSecM4Max = Math.min(m4MemTokSec, m4CompTokSec) * mtpSpeedup;

  // ==================== KV SCALING CURVE VS FRONTIER BASELINES ====================
  const ctxPoints = [
    { context: 4096, label: '4K' },
    { context: 16384, label: '16K' },
    { context: 65536, label: '64K' },
    { context: 131072, label: '128K' },
    { context: 262144, label: '256K' },
    { context: 1048576, label: '1M' },
  ];

  // Same-size MHA baseline for fair architectural comparison!
  const mhaSameSizeBytesPerTokBf16 = 2 * L * H * dHead * 2;
  const llamaGqaBytesPerTokBf16 = 128 * 1024; // Llama 3 8B: 128 KiB/tok
  const deepseekMlaBytesPerTokBf16 = 68.6 * 1024; // DeepSeek V3: 68.6 KiB/tok
  const nemotronHybridBytesPerTokBf16 = 8 * 1024; // Nemotron 3 Super: 8 KiB/tok

  const kvCurve: KvCurvePoint[] = ctxPoints.map(({ context, label }) => ({
    context,
    label,
    currentGb: Number(calcKvGbAtContext(context, kvScaleRatio).toFixed(3)),
    mhaBaselineGb: Number(
      ((mhaSameSizeBytesPerTokBf16 * context * kvScaleRatio * dims.batchSize) / 1024 ** 3).toFixed(3)
    ),
    llamaGqaGb: Number(
      ((llamaGqaBytesPerTokBf16 * context * kvScaleRatio * dims.batchSize) / 1024 ** 3).toFixed(3)
    ),
    deepseekMlaGb: Number(
      ((deepseekMlaBytesPerTokBf16 * context * kvScaleRatio * dims.batchSize) / 1024 ** 3).toFixed(3)
    ),
    nemotronHybridGb: Number(
      ((nemotronHybridBytesPerTokBf16 * context * kvScaleRatio * dims.batchSize) / 1024 ** 3).toFixed(3)
    ),
  }));

  const kvCompressionVsMha =
    kvCacheGbAtContext > 0
      ? Number(
          (
            ((mhaSameSizeBytesPerTokBf16 * dims.contextLength * kvScaleRatio * dims.batchSize) /
              1024 ** 3) /
            kvCacheGbAtContext
          ).toFixed(1)
        )
      : 999;

  // ==================== HARDWARE FIT MATRIX ====================
  const hardwareTargets = [
    { gpu: '1× RTX 4090', vramGb: 24, count: 1, bwTbs: 1.008 },
    { gpu: '1× M4 Max (128G)', vramGb: 96, count: 1, bwTbs: 0.546 },
    { gpu: '1× H100 SXM', vramGb: 80, count: 1, bwTbs: 3.35 },
    { gpu: '1× B200 (192G)', vramGb: 192, count: 1, bwTbs: 8.0 },
    { gpu: '8× H100 Node', vramGb: 640, count: 8, bwTbs: 22.0 },
  ];

  const hardwareFits: HardwareFit[] = hardwareTargets.map((hw) => {
    const fits = totalVramGb <= hw.vramGb * 0.92;
    const utilizationPct = Math.min(100, Math.round((totalVramGb / hw.vramGb) * 100));
    const estTokPerSec = fits
      ? Math.min(900, Math.round(((hw.bwTbs * 1e12 * 0.68) / Math.max(1, totalBytesRead)) * mtpSpeedup))
      : 0;
    return {
      gpu: hw.gpu,
      vramGb: hw.vramGb,
      count: hw.count,
      fits,
      utilizationPct,
      estTokPerSec,
    };
  });

  // Collect Sebastian Raschka Gallery concept tags from equipped blocks
  const conceptSet = new Set<string>();
  [tokComp, posComp, normComp, attnComp, hybComp, ffnComp, outComp].forEach((comp) => {
    comp?.concepts.forEach((tag) => conceptSet.add(tag));
  });
  const conceptTags = Array.from(conceptSet);

  // Human-readable decoder type & layer mix summary (matching Gallery fact-sheet style!)
  const decoderTypeLabel = isPureRecurrent
    ? 'Recurrent'
    : config.hybrid !== 'hyb_none' && config.hybrid !== 'hyb_swa'
      ? isMoE
        ? 'Sparse Hybrid MoE'
        : 'Dense Hybrid'
      : isMoE
        ? 'Sparse MoE'
        : 'Dense';

  const layerMixSummary = (() => {
    if (isPureRecurrent) return `${L} mLSTM`;
    const base =
      config.hybrid === 'hyb_none'
        ? `${L} ${attnComp?.shortName ?? 'GQA'}`
        : `${secondaryCount} ${hybComp?.shortName ?? 'Hybrid'} + ${primaryCount} ${attnComp?.shortName ?? 'Global'}`;
    return loopPasses > 1 ? `${base} × ${loopPasses} passes` : base;
  })();

  // ==================== SERVING CONCURRENCY & CHINCHILLA ECONOMICS ====================
  const perSeqKvGb = calcKvGbAtContext(dims.contextLength, kvScaleRatio) / Math.max(1, dims.batchSize);
  const perSeqStateGb = (recurrentStateBytes / Math.max(1, dims.batchSize)) / (1024 ** 3);
  const perSeqFootprintGb = Math.max(0.002, perSeqKvGb + perSeqStateGb + 0.015);

  const headroom1xH100 = 80 * 0.9 - weightsGb;
  const headroom8xH100 = 640 * 0.9 - weightsGb;
  const maxConcurrent1xH100 =
    headroom1xH100 > 0 ? Math.min(2048, Math.floor(headroom1xH100 / perSeqFootprintGb)) : 0;
  const maxConcurrent8xH100 =
    headroom8xH100 > 0 ? Math.min(8192, Math.floor(headroom8xH100 / perSeqFootprintGb)) : 0;

  // Chinchilla optimal tokens ≈ 20 × active params (or 20 × total for dense; MoE commonly trains ~40× active)
  const chinchillaTokens = (isMoE ? 35 : 20) * active;
  const chinchillaTokensB = chinchillaTokens / 1e9;
  const totalTrainFlops = 6 * active * chinchillaTokens;
  const trainingH100Days = totalTrainFlops / (989e12 * 0.42 * 86400);

  // Cloud serving cost per 1M generated tokens (assuming $2.60/H100-hr, batched concurrency on right-sized cluster)
  const gpusNeeded = Math.max(1, Math.ceil((weightsGb + perSeqFootprintGb * 4) / 72));
  const clusterHeadroomGb = Math.max(perSeqFootprintGb, gpusNeeded * 72 - weightsGb);
  const clusterConcurrency = Math.max(1, Math.min(256, Math.floor(clusterHeadroomGb / perSeqFootprintGb)));
  // Multi-GPU tensor parallelism scales bandwidth ~gpusNeeded^0.75, continuous batching scales throughput ~concurrency^0.65
  const effectiveClusterTokSec =
    estDecodeTokSecH100 * Math.pow(gpusNeeded, 0.72) * Math.pow(Math.min(64, clusterConcurrency), 0.6);
  const hoursPer1MTok = 1e6 / Math.max(1, effectiveClusterTokSec * 3600);
  const estServingCostPer1MTokUsd = Math.max(0.01, gpusNeeded * 2.6 * hoursPer1MTok);

  // ==================== 7-STEP LIVE TOKEN FORWARD-PASS TRACE ====================
  const totalDecodeUs = 1e6 / Math.max(1, estDecodeTokSecH100);
  const embUs = Math.max(15, Math.round(totalDecodeUs * 0.03));
  const posUs = Math.max(10, Math.round(totalDecodeUs * 0.02));
  const normUs = Math.max(18, Math.round(totalDecodeUs * (config.norm === 'norm_peri_gemma' ? 0.08 : config.norm === 'norm_mhc' ? 0.1 : 0.05)));
  const attnUs = Math.max(40, Math.round(totalDecodeUs * (isPureRecurrent ? 0.28 : 0.36)));
  const hybUs = Math.max(15, Math.round(totalDecodeUs * (config.hybrid === 'hyb_none' ? 0.04 : 0.14)));
  const ffnUs = Math.max(50, Math.round(totalDecodeUs * 0.34));
  const outUs = Math.max(25, Math.round(totalDecodeUs * 0.08));

  const traceSteps = [
    {
      step: 1,
      stage: 'tokenizer' as const,
      title: 'Tokenize & Embed Lookup',
      opSummary: `Token "attention" → ID #48291 fetched from ${V.toLocaleString()}-row table`,
      tensorIn: '[1, 1] int32',
      tensorOut: `[1, 1, ${d}] ${dims.precision.toUpperCase()}`,
      mathNote: `x₀ = E[48291] · Embed table = ${formatParams(embeddings)} params`,
      latencyUs: embUs,
      memoryReadMb: Number(((d * wBytes) / 1024 / 1024).toFixed(3)),
      highlightFact: `${tokComp?.shortName ?? 'BPE'} · V=${V.toLocaleString()}`,
    },
    {
      step: 2,
      stage: 'position' as const,
      title: 'Positional Geometry Injection',
      opSummary:
        config.position === 'pos_pure_nope'
          ? 'Pure NoPE: zero rotation applied — causal mask & recurrence carry position'
          : config.position === 'pos_abs_learned'
            ? `Adds learned position vector P[t] directly into residual stream`
            : `Rotates Q/K phase angles at position t=${formatContext(Math.min(dims.contextLength, 32768))}`,
      tensorIn: `[1, 1, ${d}]`,
      tensorOut: `[1, 1, ${d}]`,
      mathNote: posComp?.formula ?? 'R(θ, t) · q',
      latencyUs: posUs,
      memoryReadMb: config.position === 'pos_ple' ? Number(((256 * L * 2) / 1024 / 1024).toFixed(2)) : 0.01,
      highlightFact: posComp?.shortName ?? 'RoPE',
    },
    {
      step: 3,
      stage: 'norm' as const,
      title: 'Normalization & Residual Stream',
      opSummary:
        config.norm === 'norm_mhc'
          ? 'Splits token state across 4 doubly-stochastic Birkhoff residual streams'
          : config.norm === 'norm_peri_gemma'
            ? 'Applies Pre-Norm before sublayer + Post-Norm after output (4× per block)'
            : `Stabilizes hidden RMS magnitude across ${L} decoder layers`,
      tensorIn: `[1, 1, ${d}]`,
      tensorOut: config.norm === 'norm_mhc' ? `[1, 4, 1, ${d}]` : `[1, 1, ${d}]`,
      mathNote: normComp?.formula ?? 'RMSNorm(x)',
      latencyUs: normUs,
      memoryReadMb: Number(((normsAndMisc * wBytes) / 1024 / 1024).toFixed(2)),
      highlightFact: normComp?.shortName ?? 'Pre-RMSNorm',
    },
    {
      step: 4,
      stage: 'attention' as const,
      title: 'Primary Sequence Mixer & KV Read',
      opSummary: isPureRecurrent
        ? 'Bypassed — pure mLSTM matrix recurrence handles all token mixing'
        : config.attention === 'attn_mla' || config.attention === 'attn_mla_dsa' || config.attention === 'attn_csa_hca'
          ? `Compresses h → latent c_KV (${dims.mlaKvLoraRank}d) and reads ultra-lean cached latents`
          : `Projects ${H} Query heads against ${nKv} cached KV heads across ${formatContext(dims.contextLength)} context`,
      tensorIn: `[1, 1, ${d}]`,
      tensorOut: `[1, ${H}, 1, ${dHead}] → [1, 1, ${d}]`,
      mathNote: attnComp?.formula ?? 'SDPA(Q, K, V)',
      latencyUs: attnUs,
      memoryReadMb: Number((kvCacheGbAtContext * 1024 + (attentionTotal * wBytes) / 1024 / 1024).toFixed(1)),
      highlightFact: `${formatKiB(kvKiBPerTokenBf16)}/tok KV`,
    },
    {
      step: 5,
      stage: 'hybrid' as const,
      title: 'Hybrid Layer Interleave Schedule',
      opSummary:
        config.hybrid === 'hyb_none'
          ? `Uniform stack: all ${L} layers execute full ${attnComp?.shortName ?? 'attention'}`
          : config.hybrid === 'hyb_swa'
            ? `${swaCount} local layers attend strictly within W=${formatContext(dims.swaWindow)} window; ${primaryCount} layers attend globally`
            : `${secondaryCount} recurrent layers update fixed ${recurrentStateMb.toFixed(0)} MB state in O(1); ${primaryCount} layers do full recall`,
      tensorIn: `Layer Schedule (L1..L${L})`,
      tensorOut: `[1, 1, ${d}] · ${effectiveDepth} eff. layers`,
      mathNote: hybComp?.formula ?? 'Uniform Schedule',
      latencyUs: hybUs,
      memoryReadMb: Number(recurrentStateMb.toFixed(1)),
      highlightFact: layerMixSummary,
    },
    {
      step: 6,
      stage: 'ffn' as const,
      title: isMoE ? 'Sparse MoE Router & Expert Dispatch' : 'Dense Feed-Forward Nonlinearity',
      opSummary: isMoE
        ? `Router selects top-${dims.activeExperts} of ${dims.nExperts} experts${config.ffn === 'ffn_moe_shared' ? ` + ${dims.nSharedExperts} shared` : ''} (${activeRatioPct.toFixed(1)}% weights active)`
        : `Expands ${d} → ${dims.dFf.toLocaleString()} through gated nonlinearity and projects back`,
      tensorIn: `[1, 1, ${d}]`,
      tensorOut: `[1, 1, ${d}]`,
      mathNote: ffnComp?.formula ?? 'SwiGLU(x)',
      latencyUs: ffnUs,
      memoryReadMb: Number(((ffnActive * wBytes) / 1024 / 1024).toFixed(1)),
      highlightFact: `${formatParams(ffnActive)} active FFN`,
    },
    {
      step: 7,
      stage: 'output' as const,
      title: 'Final Norm, LM Head & Next-Token Sample',
      opSummary: hasMtp
        ? `Projects to ${V.toLocaleString()} logits + MTP drafts token t+2 for ~1.65× speculative speedup`
        : isTied
          ? `Reuses transposed Input Embedding Eᵀ (0 extra params) → ${V.toLocaleString()} vocab logits`
          : `Unembeds via dedicated [${d} × ${V.toLocaleString()}] matrix → next-token distribution`,
      tensorIn: `[1, 1, ${d}]`,
      tensorOut: `[1, 1, ${V.toLocaleString()}] probs`,
      mathNote: outComp?.formula ?? 'softmax(h_L · W_lmᵀ)',
      latencyUs: outUs,
      memoryReadMb: Number((((isTied ? inputEmbedParams : outputHead) * wBytes) / 1024 / 1024).toFixed(1)),
      highlightFact: `~${Math.round(estDecodeTokSecH100)} tok/s H100`,
    },
  ];

  return {
    vocabSize: V,
    dHead,
    effectiveDepth,
    params: {
      embeddings,
      attention: attentionTotal,
      ffnTotal,
      ffnActive,
      normsAndMisc,
      outputHead,
      total,
      active,
      activeRatioPct,
    },
    kvBytesPerTokenBf16: kvNominalBytesPerTokenBf16,
    kvKiBPerTokenBf16,
    kvKiBPerTokenSelected,
    kvCacheGbAtContext,
    recurrentStateMb,
    weightsGb,
    activationsGb,
    totalVramGb,
    flopsPerTokenActive,
    attentionFlopsSharePct,
    arithmeticIntensity,
    bottleneck,
    estDecodeTokSecH100,
    estDecodeTokSecM4Max,
    layers,
    kvCurve,
    hardwareFits,
    conceptTags,
    decoderTypeLabel,
    layerMixSummary,
    kvCompressionVsMha,
    economics: {
      maxConcurrent1xH100,
      maxConcurrent8xH100,
      chinchillaTokensB,
      trainingH100Days,
      estServingCostPer1MTokUsd,
    },
    traceSteps,
  };
}
