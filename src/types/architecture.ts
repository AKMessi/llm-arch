export type StageId =
  | 'tokenizer'
  | 'position'
  | 'norm'
  | 'attention'
  | 'hybrid'
  | 'ffn'
  | 'output';

export type PrecisionMode = 'fp32' | 'bf16' | 'fp8' | 'int4';

export type WorkspaceTab = 'canvas' | 'missions' | 'lab' | 'compare' | 'code';

export interface ForwardTraceStep {
  step: number;
  stage: StageId;
  title: string;
  opSummary: string;
  tensorIn: string;
  tensorOut: string;
  mathNote: string;
  latencyUs: number;
  memoryReadMb: number;
  highlightFact: string;
}

export interface ServingEconomics {
  maxConcurrent1xH100: number;
  maxConcurrent8xH100: number;
  chinchillaTokensB: number;
  trainingH100Days: number;
  estServingCostPer1MTokUsd: number;
}

export interface SavedBlueprint {
  id: string;
  savedAt: string;
  config: ArchitectureConfig;
}

export interface StageMeta {
  id: StageId;
  order: number;
  label: string;
  shortLabel: string;
  description: string;
  cssVar: string;
  softVar: string;
  tensorOut: string;
}

export interface ComponentDef {
  id: string;
  stage: StageId;
  name: string;
  shortName: string;
  badge: string;
  tagline: string;
  description: string;
  formula: string;
  pros: string[];
  tradeoffs: string[];
  examples: string[];
  concepts: string[]; // maps directly to Sebastian Raschka's related_concepts tags
  /** Optional structural overrides / hints when snapped in */
  vocabSize?: number;
  tokenizerFamily?: 'bpe_gpt2' | 'bpe_tiktoken' | 'bpe_qwen' | 'bpe_qwen_wide' | 'sp_gemma' | 'bpe_deepseek' | 'bpe_tekken' | 'byte_latent';
  kvFactorHint?: string;
  year?: number;
}

export interface ArchitectureDimensions {
  dModel: number;          // Hidden dimension, e.g. 4096
  nLayers: number;         // Total decoder layers, e.g. 32
  nHeads: number;          // Query attention heads, e.g. 32
  nKvHeads: number;        // KV heads for GQA, e.g. 8
  dFf: number;             // Dense FFN / MoE single-expert intermediate dim
  nExperts: number;        // Total routed experts (when MoE)
  activeExperts: number;   // Top-k active experts per token
  nSharedExperts: number;  // Always-on shared experts (DeepSeek/Qwen style)
  densePrefixLayers: number; // Early dense layers before MoE begins
  mlaKvLoraRank: number;   // MLA latent dim d_c (e.g. 512)
  mlaQkRopeDim: number;    // MLA decoupled RoPE dim d_r (e.g. 64)
  swaWindow: number;       // Sliding window size in tokens (e.g. 4096)
  hybridRatio: number;     // Secondary:Primary ratio (e.g. 3 means 3:1)
  loopPasses: number;      // Weight-sharing loop count (1 = standard, 2 or 4 = looped depth)
  contextLength: number;   // Target context window in tokens
  batchSize: number;       // Inference batch size for VRAM estimation
  precision: PrecisionMode;
}

export interface ArchitectureConfig {
  name: string;
  tokenizer: string;
  position: string;
  norm: string;
  attention: string;
  hybrid: string;
  ffn: string;
  output: string;
  dims: ArchitectureDimensions;
}

export interface LayerSpec {
  index: number; // 1-based
  mixerKind: 'primary' | 'swa' | 'deltanet' | 'kda' | 'mamba2' | 'mlstm' | 'shortconv';
  mixerLabel: string;
  ffnKind: 'dense' | 'moe' | 'latent_moe';
  kvBytesPerTokenBf16: number;
  isLoopedPass?: number;
  kvSharedSource?: number;
}

export interface ParamBreakdown {
  embeddings: number;
  attention: number;
  ffnTotal: number;
  ffnActive: number;
  normsAndMisc: number;
  outputHead: number;
  total: number;
  active: number;
  activeRatioPct: number;
}

export interface KvCurvePoint {
  context: number;
  label: string;
  currentGb: number;
  mhaBaselineGb: number;
  llamaGqaGb: number;
  deepseekMlaGb: number;
  nemotronHybridGb: number;
}

export interface HardwareFit {
  gpu: string;
  vramGb: number;
  count: number;
  fits: boolean;
  utilizationPct: number;
  estTokPerSec: number;
}

export interface PhysicsReport {
  vocabSize: number;
  dHead: number;
  effectiveDepth: number;
  params: ParamBreakdown;
  kvBytesPerTokenBf16: number;
  kvKiBPerTokenBf16: number;
  kvKiBPerTokenSelected: number;
  kvCacheGbAtContext: number;
  recurrentStateMb: number;
  weightsGb: number;
  activationsGb: number;
  totalVramGb: number;
  flopsPerTokenActive: number;
  attentionFlopsSharePct: number;
  arithmeticIntensity: number;
  bottleneck: 'Memory-Bandwidth Bound' | 'Compute Bound' | 'Balanced';
  estDecodeTokSecH100: number;
  estDecodeTokSecM4Max: number;
  layers: LayerSpec[];
  kvCurve: KvCurvePoint[];
  hardwareFits: HardwareFit[];
  conceptTags: string[];
  decoderTypeLabel: string;
  layerMixSummary: string;
  kvCompressionVsMha: number;
  economics: ServingEconomics;
  traceSteps: ForwardTraceStep[];
}

export interface DiagnosticItem {
  id: string;
  severity: 'synergy' | 'info' | 'warning' | 'critical';
  title: string;
  detail: string;
  suggestion?: {
    label: string;
    stage: StageId;
    componentId: string;
  };
}

export interface GalleryModel {
  id: string;
  name: string;
  company: string;
  country: string;
  date: string;
  summary: string;
  highlight: string;
  scale: string;
  totalParamsB: number | null;
  activeParamsB: number | null;
  contextTokensRaw: string;
  contextTokens: number | null;
  vocabSizeRaw: string;
  vocabSize: number | null;
  licenseName: string;
  decoderType: string;
  attention: string;
  layerMix: string;
  kvCachePerTokenRaw: string;
  kvCacheKiB: number | null;
  concepts: string[];
  imageUrl: string;
  galleryUrl: string;
  configUrl: string;
  configRepo: string;
  techReportUrl: string;
}

export interface SimilarityMatch {
  model: GalleryModel;
  score: number; // 0..100
  sharedConcepts: string[];
  missingInUser: string[];
  extraInUser: string[];
}

export interface PresetBlueprint {
  id: string;
  name: string;
  company: string;
  year: string;
  tagline: string;
  scaleBadge: string;
  kvBadge: string;
  galleryModelName: string;
  config: ArchitectureConfig;
}
