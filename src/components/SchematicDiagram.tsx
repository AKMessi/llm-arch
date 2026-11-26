import React, { useRef, useState } from 'react';
import { Download, Image as ImageIcon, Play, Pause, Maximize2 } from 'lucide-react';
import { COMPONENT_MAP } from '../data/components';
import { formatContext, formatKiB, formatParams } from '../engine/physics';
import type { ArchitectureConfig, PhysicsReport, StageId } from '../types/architecture';

interface SchematicDiagramProps {
  config: ArchitectureConfig;
  physics: PhysicsReport;
  theme: 'paper' | 'obsidian';
  onSelectStage?: (stage: StageId) => void;
  highlightedStage?: StageId | null;
}

export const SchematicDiagram: React.FC<SchematicDiagramProps> = ({
  config,
  physics,
  theme,
  onSelectStage,
  highlightedStage,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [animatePulse, setAnimatePulse] = useState(true);
  const [expanded, setExpanded] = useState(false);

  const tok = COMPONENT_MAP[config.tokenizer];
  const pos = COMPONENT_MAP[config.position];
  const norm = COMPONENT_MAP[config.norm];
  const attn = COMPONENT_MAP[config.attention];
  const hyb = COMPONENT_MAP[config.hybrid];
  const ffn = COMPONENT_MAP[config.ffn];
  const out = COMPONENT_MAP[config.output];

  const { dims } = config;
  const isPaper = theme === 'paper';

  const ink = isPaper ? '#15171c' : '#f1f4f9';
  const inkSec = isPaper ? '#4a4f5c' : '#a5b0c4';
  const inkMuted = isPaper ? '#767c8a' : '#7a879f';
  const surface = isPaper ? '#ffffff' : '#141821';
  const surfaceSub = isPaper ? '#f6f3ec' : '#1a202c';
  const border = isPaper ? '#d4cdc0' : '#2e374a';
  const borderStrong = isPaper ? '#8f887b' : '#4a5670';

  const cTok = isPaper ? '#2563eb' : '#60a5fa';
  const cPos = isPaper ? '#4f46e5' : '#818cf8';
  const cNorm = isPaper ? '#0d9488' : '#2dd4bf';
  const cAttn = isPaper ? '#ea580c' : '#fb923c';
  const cHyb = isPaper ? '#d97706' : '#fbbf24';
  const cFfn = isPaper ? '#059669' : '#34d399';
  const cOut = isPaper ? '#e11d48' : '#fb7185';

  const isPreNorm =
    config.norm === 'norm_pre_ln' ||
    config.norm === 'norm_pre_rms' ||
    config.norm === 'norm_rms_qknorm' ||
    config.norm === 'norm_mhc' ||
    config.norm === 'norm_attn_res' ||
    config.norm === 'norm_gated_loop';
  const isPeriNorm = config.norm === 'norm_peri_gemma';
  const isPostNorm = config.norm === 'norm_post_olmo';
  const isMhc = config.norm === 'norm_mhc';
  const isAttnRes = config.norm === 'norm_attn_res';
  const isGatedLoopNorm = config.norm === 'norm_gated_loop';
  const hasQkNorm =
    config.norm === 'norm_rms_qknorm' ||
    config.norm === 'norm_peri_gemma' ||
    config.norm === 'norm_post_olmo';
  const normShort = config.norm === 'norm_pre_ln' ? 'LayerNorm' : 'RMSNorm';

  const isMla =
    config.attention === 'attn_mla' ||
    config.attention === 'attn_mla_dsa' ||
    config.attention === 'attn_mla_dsa_loop' ||
    config.attention === 'attn_gated_mla' ||
    config.attention === 'attn_csa_hca';
  const isGatedAttn =
    config.attention === 'attn_gated_gqa' ||
    config.attention === 'attn_gated_mla' ||
    config.attention === 'attn_difftpa';
  const isHybrid = config.hybrid !== 'hyb_none';
  const isPureRecurrent = config.hybrid === 'hyb_xlstm';
  const isMoE =
    config.ffn === 'ffn_moe_sparse' ||
    config.ffn === 'ffn_moe_shared' ||
    config.ffn === 'ffn_latent_moe' ||
    config.ffn === 'ffn_looped_moe' ||
    config.ffn === 'ffn_ngram_moe';
  const isTied = config.output === 'out_tied' || config.output === 'out_softcap';
  const hasMtp = config.output === 'out_mtp' || config.output === 'out_sinks_mtp';
  const hasSoftCap = config.output === 'out_softcap';
  const isLooped = config.ffn === 'ffn_looped_depth' || config.ffn === 'ffn_looped_moe';
  const isSandwichLoop = config.ffn === 'ffn_looped_moe';
  const hasPle = config.position === 'pos_ple';

  const nKv =
    config.attention === 'attn_mha'
      ? dims.nHeads
      : config.attention === 'attn_mqa'
        ? 1
        : dims.nKvHeads;

  const exportSvg = () => {
    if (!svgRef.current) return;
    const clone = svgRef.current.cloneNode(true) as SVGSVGElement;
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bgRect.setAttribute('width', '100%');
    bgRect.setAttribute('height', '100%');
    bgRect.setAttribute('fill', isPaper ? '#f8f6f0' : '#0e1118');
    clone.insertBefore(bgRect, clone.firstChild);

    const blob = new Blob([new XMLSerializer().serializeToString(clone)], {
      type: 'image/svg+xml;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${config.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-blueprint.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportPng = () => {
    if (!svgRef.current) return;
    const clone = svgRef.current.cloneNode(true) as SVGSVGElement;
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bgRect.setAttribute('width', '100%');
    bgRect.setAttribute('height', '100%');
    bgRect.setAttribute('fill', isPaper ? '#f8f6f0' : '#0e1118');
    clone.insertBefore(bgRect, clone.firstChild);

    const svgStr = new XMLSerializer().serializeToString(clone);
    const img = new Image();
    const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 960;
      canvas.height = 1480;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const pngUrl = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = pngUrl;
        a.download = `${config.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-blueprint.png`;
        a.click();
      }
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const renderSvg = (className: string) => (
    <svg
      ref={svgRef}
      viewBox="0 0 480 740"
      className={className}
      style={{ fontFamily: "'Inter', system-ui, sans-serif" }}
    >
      <defs>
        <marker id="arrowInk" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1 L 7 4 L 0 7 z" fill={borderStrong} />
        </marker>
        <marker id="arrowOut" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1 L 7 4 L 0 7 z" fill={cOut} />
        </marker>
        <marker id="arrowNorm" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1 L 7 4 L 0 7 z" fill={cNorm} />
        </marker>
      </defs>

      {/* Blueprint Title Stamp */}
      <g>
        <text x="20" y="22" fontSize="9.5" fontWeight="700" fill={inkMuted} letterSpacing="1.2">
          SCHEMATIC FIG. 01 · {physics.decoderTypeLabel.toUpperCase()}
        </text>
        <text x="460" y="22" textAnchor="end" fontSize="9.5" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
          {formatParams(physics.params.total)} ({formatParams(physics.params.active)} ACT) · {formatKiB(physics.kvKiBPerTokenBf16)}/TOK
        </text>
        <line x1="20" y1="29" x2="460" y2="29" stroke={border} strokeWidth="1" />
      </g>

      {/* ==================== STAGE 1: TOKENIZER & EMBEDDING ==================== */}
      <g
        className="cursor-pointer"
        onClick={() => onSelectStage?.('tokenizer')}
      >
        <rect
          x="90"
          y="42"
          width="300"
          height="42"
          rx="6"
          fill={surface}
          stroke={highlightedStage === 'tokenizer' ? cTok : border}
          strokeWidth={highlightedStage === 'tokenizer' ? 2 : 1.2}
        />
        <rect x="90" y="42" width="4" height="42" rx="2" fill={cTok} />
        <text x="104" y="59" fontSize="11" fontWeight="700" fill={ink}>
          {tok?.shortName} · Token Embedding
        </text>
        <text x="104" y="74" fontSize="9.5" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
          V={physics.vocabSize.toLocaleString()} → [B, T, {dims.dModel}]
        </text>
        <rect x="330" y="51" width="50" height="17" rx="3" fill={isPaper ? 'rgba(37,99,235,0.1)' : 'rgba(96,165,250,0.15)'} />
        <text x="355" y="63" textAnchor="middle" fontSize="8.5" fontWeight="700" fontFamily="'JetBrains Mono', monospace" fill={cTok}>
          EMBED
        </text>
      </g>

      {/* Spine Connector 1 -> 2 */}
      <line
        x1="240"
        y1="84"
        x2="240"
        y2="106"
        stroke={borderStrong}
        strokeWidth="1.5"
        markerEnd="url(#arrowInk)"
        className={animatePulse ? 'animate-tensor-flow' : ''}
      />

      {/* ==================== STAGE 2: POSITIONAL / PLE INJECTION ==================== */}
      <g
        className="cursor-pointer"
        onClick={() => onSelectStage?.('position')}
      >
        <rect
          x="110"
          y="108"
          width="260"
          height="34"
          rx="5"
          fill={surface}
          stroke={highlightedStage === 'position' ? cPos : border}
          strokeWidth={highlightedStage === 'position' ? 2 : 1.2}
        />
        <rect x="110" y="108" width="4" height="34" rx="2" fill={cPos} />
        <text x="124" y="124" fontSize="10.5" fontWeight="700" fill={ink}>
          {pos?.shortName}
        </text>
        <text x="124" y="136" fontSize="9" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
          {config.position === 'pos_abs_learned'
            ? `x₀ + P[0..${formatContext(dims.contextLength)}]`
            : config.position === 'pos_pure_nope'
              ? 'Implicit causal position (NoPE)'
              : hasPle
                ? 'RoPE + Per-Layer Token Embeds'
                : `Context Window: ${formatContext(dims.contextLength)} tok`}
        </text>
        <text x="360" y="128" textAnchor="end" fontSize="8.5" fontWeight="700" fontFamily="'JetBrains Mono', monospace" fill={cPos}>
          POS
        </text>
      </g>

      {/* Spine Connector 2 -> Decoder Block */}
      <line
        x1="240"
        y1="142"
        x2="240"
        y2={isPreNorm || isPeriNorm ? 190 : 202}
        stroke={borderStrong}
        strokeWidth="1.5"
        markerEnd="url(#arrowInk)"
        className={animatePulse ? 'animate-tensor-flow' : ''}
      />

      {/* ==================== REPEATING DECODER BLOCK FRAME ==================== */}
      <g>
        <rect
          x="26"
          y="162"
          width="428"
          height="422"
          rx="10"
          fill={surfaceSub}
          stroke={borderStrong}
          strokeWidth="1.3"
          strokeDasharray="6 4"
        />
        {/* Block Stack Header Pill */}
        <rect x="38" y="152" width={isSandwichLoop ? 212 : 166} height="20" rx="4" fill={ink} />
        <text x="48" y="165.5" fontSize="9.2" fontWeight="700" fontFamily="'JetBrains Mono', monospace" fill={isPaper ? '#fff' : '#0e1118'}>
          {isSandwichLoop && physics.loopBreakdown
            ? `LOOP ${physics.loopBreakdown.opening}+${physics.loopBreakdown.looped}×${physics.loopBreakdown.passes}+${physics.loopBreakdown.closing} (${physics.effectiveDepth} EFF)`
            : `DECODER BLOCK × ${dims.nLayers}${isLooped ? ` × ${dims.loopPasses} LOOPS` : ''}`}
        </text>
        <rect x={isSandwichLoop ? 264 : 282} y="152" width={isSandwichLoop ? 178 : 160} height="20" rx="4" fill={surface} stroke={border} />
        <text x={isSandwichLoop ? 353 : 362} y="165.5" textAnchor="middle" fontSize="8.3" fontWeight="600" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
          {norm?.shortName} · {physics.layerMixSummary.slice(0, 18)}
        </text>

        {/* Per-Layer Embedding Side Injection Arrow */}
        {hasPle && (
          <g>
            <path d="M 390 63 L 444 63 L 444 480 L 424 480" fill="none" stroke={cPos} strokeWidth="1.2" strokeDasharray="3 3" />
            <text x="441" y="290" fontSize="8" fontWeight="700" fill={cPos} transform="rotate(-90 441 290)" textAnchor="middle">
              PER-LAYER EMBEDDING (PLE)
            </text>
          </g>
        )}

        {/* ==================== RESIDUAL HIGHWAY (LEFT RAIL) ==================== */}
        <g
          className="cursor-pointer"
          onClick={() => onSelectStage?.('norm')}
        >
          {isMhc ? (
            <>
              {/* 4 Parallel Birkhoff mHC Residual Streams with horizontal taps */}
              <path d="M 240 184 L 44 184" fill="none" stroke={cNorm} strokeWidth="1.3" />
              <path d="M 62 378 L 228 378" fill="none" stroke={cNorm} strokeWidth="1.5" />
              <path d="M 62 560 L 228 560" fill="none" stroke={cNorm} strokeWidth="1.5" />
              {[44, 50, 56, 62].map((rx, idx) => (
                <path
                  key={rx}
                  d={`M ${rx} 184 L ${rx} 560`}
                  fill="none"
                  stroke={cNorm}
                  strokeWidth={idx === 1 || idx === 2 ? 1.5 : 1.1}
                  opacity={0.85}
                  className={animatePulse ? 'animate-tensor-flow' : ''}
                />
              ))}
              <rect x="36" y="352" width="35" height="32" rx="4" fill={surface} stroke={cNorm} strokeWidth="1.3" />
              <text x="53.5" y="366" textAnchor="middle" fontSize="8" fontWeight="700" fill={cNorm}>
                mHC
              </text>
              <text x="53.5" y="377" textAnchor="middle" fontSize="7" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                4-lane
              </text>
            </>
          ) : isAttnRes ? (
            <>
              <path d="M 240 184 L 52 184" fill="none" stroke={cNorm} strokeWidth="1.4" />
              <path d="M 52 378 L 228 378" fill="none" stroke={cNorm} strokeWidth="1.5" />
              <path d="M 52 560 L 228 560" fill="none" stroke={cNorm} strokeWidth="1.5" />
              <path
                d="M 52 184 L 52 560"
                fill="none"
                stroke={cNorm}
                strokeWidth="2"
                className={animatePulse ? 'animate-tensor-flow' : ''}
              />
              <path d="M 52 210 Q 32 310 52 378" fill="none" stroke={cNorm} strokeWidth="1.2" strokeDasharray="3 2" />
              <path d="M 52 210 Q 26 400 52 560" fill="none" stroke={cNorm} strokeWidth="1.2" strokeDasharray="3 2" />
              <rect x="33" y="355" width="38" height="22" rx="4" fill={surface} stroke={cNorm} strokeWidth="1.2" />
              <text x="52" y="369" textAnchor="middle" fontSize="7.5" fontWeight="700" fill={cNorm}>
                α·Res
              </text>
            </>
          ) : (
            <>
              {/* Classic Clean Residual Bypass Rail */}
              <path
                d="M 240 184 L 54 184 L 54 378 L 228 378"
                fill="none"
                stroke={cNorm}
                strokeWidth="1.5"
                className={animatePulse ? 'animate-tensor-flow' : ''}
              />
              <path
                d="M 240 392 L 54 392 L 54 560 L 228 560"
                fill="none"
                stroke={cNorm}
                strokeWidth="1.5"
                className={animatePulse ? 'animate-tensor-flow' : ''}
              />
              <text x="46" y="285" fontSize="8" fontWeight="600" fill={cNorm} transform="rotate(-90 46 285)" textAnchor="middle">
                RESIDUAL
              </text>
              <text x="46" y="476" fontSize="8" fontWeight="600" fill={cNorm} transform="rotate(-90 46 476)" textAnchor="middle">
                RESIDUAL
              </text>
            </>
          )}
        </g>

        {/* ==================== SUB-BLOCK 1: NORM 1 (PRE / PERI) ==================== */}
        {(isPreNorm || isPeriNorm) && (
          <g className="cursor-pointer" onClick={() => onSelectStage?.('norm')}>
            <rect
              x="160"
              y="192"
              width="160"
              height="22"
              rx="4"
              fill={surface}
              stroke={highlightedStage === 'norm' ? cNorm : cNorm}
              strokeWidth={highlightedStage === 'norm' ? 2 : 1.2}
            />
            <text x="240" y="206.5" textAnchor="middle" fontSize="9.5" fontWeight="700" fill={cNorm}>
              Pre-{normShort} {isMhc ? '· H_pre' : isGatedLoopNorm ? '· β_k Gate' : ''}
            </text>
            <line x1="240" y1="214" x2="240" y2="226" stroke={borderStrong} strokeWidth="1.3" />
          </g>
        )}

        {/* ==================== SUB-BLOCK 2: ATTENTION / HYBRID SEQUENCE MIXER ==================== */}
        <g
          className="cursor-pointer"
          onClick={() => onSelectStage?.(isHybrid ? 'hybrid' : 'attention')}
        >
          <rect
            x="84"
            y={isPreNorm || isPeriNorm ? 226 : 204}
            width="342"
            height={isPreNorm || isPeriNorm ? 110 : 124}
            rx="7"
            fill={surface}
            stroke={
              highlightedStage === 'attention'
                ? cAttn
                : highlightedStage === 'hybrid'
                  ? cHyb
                  : cAttn
            }
            strokeWidth={
              highlightedStage === 'attention' || highlightedStage === 'hybrid' ? 2.2 : 1.4
            }
          />

          {/* Top Bar inside Attention Box */}
          <rect
            x="84"
            y={isPreNorm || isPeriNorm ? 226 : 204}
            width="342"
            height="22"
            rx="6"
            fill={isPaper ? 'rgba(234,88,12,0.09)' : 'rgba(251,146,60,0.14)'}
          />
          <text
            x="96"
            y={(isPreNorm || isPeriNorm ? 226 : 204) + 14.5}
            fontSize="10"
            fontWeight="700"
            fill={cAttn}
          >
            {isPureRecurrent
              ? 'mLSTM MATRIX RECURRENCE (ATTENTION-FREE)'
              : isHybrid
                ? `HYBRID MIXER · ${dims.hybridRatio}:1 ${hyb?.shortName.toUpperCase()} + ${attn?.shortName.toUpperCase()}`
                : `${attn?.name.toUpperCase()}`}
          </text>

          {/* Dynamic Internals of Sequence Mixer */}
          {isPureRecurrent ? (
            <g transform={`translate(0, ${isPreNorm || isPeriNorm ? 0 : -16})`}>
              <rect x="104" y="258" width="142" height="58" rx="5" fill={surfaceSub} stroke={cHyb} />
              <text x="175" y="278" textAnchor="middle" fontSize="10" fontWeight="700" fill={ink}>
                Exp-Gated Memory C_t
              </text>
              <text x="175" y="294" textAnchor="middle" fontSize="8.5" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
                C_t = f_t·C_(t-1) + i_t·v·kᵀ
              </text>
              <text x="175" y="308" textAnchor="middle" fontSize="8.5" fontWeight="700" fill={cFfn}>
                Fixed State: {physics.recurrentStateMb.toFixed(0)} MB
              </text>

              <rect x="262" y="258" width="146" height="58" rx="5" fill={surfaceSub} stroke={border} />
              <text x="335" y="281" textAnchor="middle" fontSize="10" fontWeight="700" fill={cFfn}>
                0 B / token KV Cache
              </text>
              <text x="335" y="298" textAnchor="middle" fontSize="8.5" fill={inkMuted}>
                O(1) Memory · Infinite Ctx
              </text>
            </g>
          ) : isHybrid ? (
            /* Dual-Track Hybrid View! */
            <g transform={`translate(0, ${isPreNorm || isPeriNorm ? 0 : -14})`}>
              {/* Left Track: Secondary Hybrid Mixer */}
              <rect x="96" y="256" width="156" height="68" rx="5" fill={surfaceSub} stroke={cHyb} strokeWidth="1.2" />
              <text x="106" y="272" fontSize="9.5" fontWeight="700" fill={cHyb}>
                {dims.hybridRatio}× {hyb?.shortName}
              </text>
              <text x="106" y="286" fontSize="8.3" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
                {config.hybrid === 'hyb_swa'
                  ? `Local Window W=${formatContext(dims.swaWindow)}`
                  : config.hybrid === 'hyb_mamba3'
                    ? 'Mamba-3 Complex SSM'
                    : config.hybrid === 'hyb_mamba2'
                      ? 'State-Space Duality (O(1))'
                      : config.hybrid === 'hyb_shortconv'
                        ? 'Causal Conv1D + Gate'
                        : 'Delta Rule + ShortConv'}
              </text>
              <rect x="106" y="296" width="136" height="19" rx="3" fill={isPaper ? 'rgba(217,119,6,0.12)' : 'rgba(251,191,36,0.16)'} />
              <text x="174" y="308.5" textAnchor="middle" fontSize="8" fontWeight="700" fontFamily="'JetBrains Mono', monospace" fill={cHyb}>
                {config.hybrid === 'hyb_swa' ? 'KV CAPPED AT W' : 'ZERO KV CACHE · O(1)'}
              </text>

              {/* Right Track: Global Primary Attention */}
              <rect x="262" y="256" width="152" height="68" rx="5" fill={surfaceSub} stroke={cAttn} strokeWidth="1.2" />
              <text x="272" y="272" fontSize="9.5" fontWeight="700" fill={cAttn}>
                1× {attn?.shortName} (Global)
              </text>
              <text x="272" y="286" fontSize="8.3" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
                {isMla
                  ? `Latent c_KV=${dims.mlaKvLoraRank}d`
                  : `${dims.nHeads}Q : ${nKv}KV · d_h=${physics.dHead}`}
              </text>
              <g>
                {hasQkNorm && (
                  <>
                    <rect x="272" y="296" width="54" height="18" rx="3" fill={isPaper ? 'rgba(13,148,136,0.12)' : 'rgba(45,212,191,0.15)'} />
                    <text x="299" y="308" textAnchor="middle" fontSize="7.8" fontWeight="700" fill={cNorm}>QK-Norm</text>
                  </>
                )}
                {isGatedAttn && (
                  <>
                    <rect x={hasQkNorm ? 330 : 272} y="296" width="54" height="18" rx="3" fill={isPaper ? 'rgba(234,88,12,0.12)' : 'rgba(251,146,60,0.16)'} />
                    <text x={hasQkNorm ? 357 : 299} y="308" textAnchor="middle" fontSize="7.8" fontWeight="700" fill={cAttn}>Gate σ</text>
                  </>
                )}
                {!hasQkNorm && !isGatedAttn && (
                  <text x="272" y="309" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                    Full Global Recall
                  </text>
                )}
              </g>
            </g>
          ) : isMla ? (
            /* MLA Low-Rank Latent Bottleneck Schematic! */
            <g transform={`translate(0, ${isPreNorm || isPeriNorm ? 0 : -14})`}>
              <rect x="98" y="258" width="72" height="58" rx="4" fill={surfaceSub} stroke={border} />
              <text x="134" y="280" textAnchor="middle" fontSize="9.5" fontWeight="700" fill={ink}>Q Path</text>
              <text x="134" y="295" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
                {dims.nHeads}H · RoPE
              </text>

              <line x1="170" y1="287" x2="188" y2="287" stroke={borderStrong} strokeWidth="1.2" />

              {/* Compressed Latent Cache Node */}
              <rect x="188" y="256" width="126" height="62" rx="5" fill={isPaper ? 'rgba(234,88,12,0.08)' : 'rgba(251,146,60,0.13)'} stroke={cAttn} strokeWidth="1.4" />
              <text x="251" y="272" textAnchor="middle" fontSize="8.6" fontWeight="700" fill={cAttn}>
                {config.attention === 'attn_mla_dsa_loop'
                  ? 'HYBRID LOOP KV'
                  : config.attention === 'attn_gated_mla'
                    ? 'GATED MLA · IDX'
                    : 'LATENT KV CACHE'}
              </text>
              <text x="251" y="287" textAnchor="middle" fontSize="8.5" fontFamily="'JetBrains Mono', monospace" fill={ink}>
                c_KV({dims.mlaKvLoraRank}) ⊕ k_R({dims.mlaQkRopeDim})
              </text>
              <text x="251" y="304" textAnchor="middle" fontSize="8.2" fontWeight="700" fontFamily="'JetBrains Mono', monospace" fill={cFfn}>
                {config.attention === 'attn_mla_dsa_loop'
                  ? `P2: SWA(${formatContext(dims.swaWindow)})`
                  : `${formatKiB(physics.kvKiBPerTokenBf16)}/tok (-${Math.min(99, Math.round((1 - 1 / Math.max(1, physics.kvCompressionVsMha)) * 100))}%)`}
              </text>

              <line x1="314" y1="287" x2="332" y2="287" stroke={borderStrong} strokeWidth="1.2" />

              <rect x="332" y="258" width="80" height="58" rx="4" fill={surfaceSub} stroke={border} />
              <text x="372" y="280" textAnchor="middle" fontSize="8.8" fontWeight="700" fill={ink}>
                {config.attention === 'attn_mla_dsa_loop'
                  ? 'IdxShare'
                  : config.attention === 'attn_gated_mla'
                    ? 'Gate σ'
                    : config.attention === 'attn_mla_dsa' || config.attention === 'attn_csa_hca'
                      ? 'Top-k DSA'
                      : 'Absorbed'}
              </text>
              <text x="372" y="295" textAnchor="middle" fontSize="7.8" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
                {config.attention === 'attn_mla_dsa_loop'
                  ? 'DSA Reuse'
                  : config.attention === 'attn_gated_mla'
                    ? 'IndexCache'
                    : config.attention === 'attn_csa_hca'
                      ? 'CSA/HCA'
                      : 'W_UKV · W_O'}
              </text>
            </g>
          ) : (
            /* Standard MHA / GQA / MQA / Gated GQA QKV Flow */
            <g transform={`translate(0, ${isPreNorm || isPeriNorm ? 0 : -14})`}>
              {[
                { label: 'Q', x: 100, sub: `${dims.nHeads}H`, normPill: hasQkNorm },
                { label: 'K', x: 184, sub: `${nKv}H`, normPill: hasQkNorm },
                { label: 'V', x: 268, sub: `${nKv}H`, normPill: false },
              ].map((p) => (
                <g key={p.label}>
                  <rect x={p.x} y="256" width="68" height="36" rx="4" fill={surfaceSub} stroke={border} />
                  <text x={p.x + 34} y="271" textAnchor="middle" fontSize="10" fontWeight="700" fill={ink}>
                    {p.label} ({p.sub})
                  </text>
                  <text x={p.x + 34} y="284" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                    d_h={physics.dHead}
                  </text>
                  {p.normPill && (
                    <g>
                      <rect x={p.x + 5} y="297" width="58" height="16" rx="3" fill={isPaper ? 'rgba(13,148,136,0.13)' : 'rgba(45,212,191,0.16)'} />
                      <text x={p.x + 34} y="308" textAnchor="middle" fontSize="7.5" fontWeight="700" fill={cNorm}>
                        QK-Norm
                      </text>
                    </g>
                  )}
                </g>
              ))}

              {/* Output Projection / Gate Box */}
              <rect x="350" y="256" width="64" height="58" rx="4" fill={isPaper ? 'rgba(234,88,12,0.08)' : 'rgba(251,146,60,0.14)'} stroke={cAttn} />
              <text x="382" y="276" textAnchor="middle" fontSize="8.8" fontWeight="700" fill={cAttn}>
                {config.attention === 'attn_difftpa'
                  ? 'DiffTPA r4'
                  : isGatedAttn
                    ? '⊙ σ(W_g)'
                    : config.attention === 'attn_kv_sharing'
                      ? 'KV-Share'
                      : 'SDPA'}
              </text>
              <text x="382" y="290" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
                → W_O
              </text>
              <text x="382" y="305" textAnchor="middle" fontSize="7.5" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                {formatKiB(physics.kvKiBPerTokenBf16)}
              </text>
            </g>
          )}
        </g>

        {/* Post-Attn Norm (Peri-Norm or Post-Norm) */}
        {(isPeriNorm || isPostNorm) && (
          <g className="cursor-pointer" onClick={() => onSelectStage?.('norm')}>
            <line x1="240" y1={isPeriNorm ? 336 : 328} x2="240" y2="344" stroke={borderStrong} strokeWidth="1.3" />
            <rect x="160" y="344" width="160" height="20" rx="4" fill={surface} stroke={cNorm} strokeWidth="1.2" />
            <text x="240" y="357.5" textAnchor="middle" fontSize="9" fontWeight="700" fill={cNorm}>
              Post-Attn {normShort}
            </text>
          </g>
        )}

        {/* Residual Addition Node #1 */}
        <line x1="240" y1={isPeriNorm || isPostNorm ? 364 : 336} x2="240" y2="370" stroke={borderStrong} strokeWidth="1.4" />
        <circle cx="240" cy="378" r="10" fill={surface} stroke={cNorm} strokeWidth="1.8" />
        <text x="240" y="382" textAnchor="middle" fontSize="13" fontWeight="700" fill={cNorm}>+</text>
        <line x1="240" y1="388" x2="240" y2="404" stroke={borderStrong} strokeWidth="1.4" />

        {/* ==================== SUB-BLOCK 3: NORM 2 (PRE / PERI) ==================== */}
        {(isPreNorm || isPeriNorm) && (
          <g className="cursor-pointer" onClick={() => onSelectStage?.('norm')}>
            <rect x="160" y="404" width="160" height="22" rx="4" fill={surface} stroke={cNorm} strokeWidth="1.2" />
            <text x="240" y="418.5" textAnchor="middle" fontSize="9.5" fontWeight="700" fill={cNorm}>
              Pre-FFN {normShort}
            </text>
            <line x1="240" y1="426" x2="240" y2="436" stroke={borderStrong} strokeWidth="1.3" />
          </g>
        )}

        {/* ==================== SUB-BLOCK 4: FFN / SPARSE MoE ==================== */}
        <g
          className="cursor-pointer"
          onClick={() => onSelectStage?.('ffn')}
        >
          <rect
            x="84"
            y={isPreNorm || isPeriNorm ? 436 : 410}
            width="342"
            height={isPreNorm || isPeriNorm ? 86 : 102}
            rx="7"
            fill={surface}
            stroke={cFfn}
            strokeWidth={highlightedStage === 'ffn' ? 2.2 : 1.4}
          />
          <rect
            x="84"
            y={isPreNorm || isPeriNorm ? 436 : 410}
            width="342"
            height="21"
            rx="6"
            fill={isPaper ? 'rgba(5,150,105,0.09)' : 'rgba(52,211,153,0.14)'}
          />
          <text
            x="96"
            y={(isPreNorm || isPeriNorm ? 436 : 410) + 14}
            fontSize="10"
            fontWeight="700"
            fill={cFfn}
          >
            {ffn?.name.toUpperCase()}
          </text>
          <text
            x="414"
            y={(isPreNorm || isPeriNorm ? 436 : 410) + 14}
            textAnchor="end"
            fontSize="8.5"
            fontFamily="'JetBrains Mono', monospace"
            fill={inkSec}
          >
            {isMoE ? `${physics.params.activeRatioPct.toFixed(1)}% ACTIVE` : `d_ff=${dims.dFf.toLocaleString()}`}
          </text>

          {isMoE ? (
            <g transform={`translate(0, ${isPreNorm || isPeriNorm ? 0 : -14})`}>
              {/* Router Box */}
              <rect x="96" y="466" width="68" height="44" rx="4" fill={surfaceSub} stroke={border} />
              <text x="130" y="484" textAnchor="middle" fontSize="9" fontWeight="700" fill={ink}>
                Router
              </text>
              <text x="130" y="498" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={cFfn}>
                Top-{dims.activeExperts}/{dims.nExperts}
              </text>

              {/* Expert Cells Grid */}
              <g>
                {[0, 1, 2, 3, 4, 5, 6, 7].map((idx) => {
                  const activeCell = idx < Math.min(8, Math.max(2, Math.round((dims.activeExperts / Math.max(1, dims.nExperts)) * 24)));
                  const cx = 178 + (idx % 4) * 38;
                  const cy = 466 + Math.floor(idx / 4) * 24;
                  return (
                    <g key={idx}>
                      <rect
                        x={cx}
                        y={cy}
                        width="33"
                        height="19"
                        rx="3"
                        fill={activeCell ? (isPaper ? 'rgba(5,150,105,0.16)' : 'rgba(52,211,153,0.22)') : surfaceSub}
                        stroke={activeCell ? cFfn : border}
                        strokeWidth={activeCell ? 1.3 : 1}
                      />
                      <text
                        x={cx + 16.5}
                        y={cy + 12.5}
                        textAnchor="middle"
                        fontSize="7.5"
                        fontWeight={activeCell ? '700' : '500'}
                        fontFamily="'JetBrains Mono', monospace"
                        fill={activeCell ? cFfn : inkMuted}
                      >
                        E{idx + 1}
                      </text>
                    </g>
                  );
                })}
              </g>

              {/* Shared Expert, N-Gram Table, or Latent Bottleneck Box */}
              <rect x="338" y="466" width="76" height="44" rx="4" fill={surfaceSub} stroke={config.ffn === 'ffn_moe_shared' || config.ffn === 'ffn_looped_moe' || config.ffn === 'ffn_latent_moe' || config.ffn === 'ffn_ngram_moe' ? cFfn : border} strokeDasharray={config.ffn === 'ffn_moe_sparse' ? '3 2' : undefined} />
              <text x="376" y="484" textAnchor="middle" fontSize="8.5" fontWeight="700" fill={config.ffn === 'ffn_moe_sparse' ? inkMuted : cFfn}>
                {config.ffn === 'ffn_latent_moe'
                  ? 'Latent z'
                  : config.ffn === 'ffn_ngram_moe'
                    ? 'N-Gram LUT'
                    : config.ffn === 'ffn_moe_shared' || config.ffn === 'ffn_looped_moe'
                      ? `${dims.nSharedExperts}× Shared`
                      : 'No Shared'}
              </text>
              <text x="376" y="498" textAnchor="middle" fontSize="7.5" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                {config.ffn === 'ffn_latent_moe'
                  ? `d/2=${Math.round(dims.dModel / 2)}`
                  : config.ffn === 'ffn_ngram_moe'
                    ? 'O(1) Engram'
                    : isSandwichLoop && physics.loopBreakdown
                      ? `${physics.loopBreakdown.looped}L × ${physics.loopBreakdown.passes} Loop`
                      : dims.densePrefixLayers > 0
                        ? `${dims.densePrefixLayers}L Dense Pre`
                        : 'All-MoE'}
              </text>
            </g>
          ) : (
            /* Dense SwiGLU / GeGLU / GELU / Looped Depth */
            <g transform={`translate(0, ${isPreNorm || isPeriNorm ? 0 : -14})`}>
              <rect x="102" y="468" width="90" height="40" rx="4" fill={surfaceSub} stroke={border} />
              <text x="147" y="485" textAnchor="middle" fontSize="9" fontWeight="700" fill={ink}>
                {config.ffn === 'ffn_gelu' ? 'W_up (4×)' : 'W_gate ⊙ W_up'}
              </text>
              <text x="147" y="499" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                {dims.dModel} → {dims.dFf}
              </text>

              <line x1="192" y1="488" x2="212" y2="488" stroke={borderStrong} strokeWidth="1.2" />

              <rect x="212" y="472" width="76" height="32" rx="4" fill={isPaper ? 'rgba(5,150,105,0.12)' : 'rgba(52,211,153,0.16)'} stroke={cFfn} />
              <text x="250" y="491" textAnchor="middle" fontSize="9.5" fontWeight="700" fill={cFfn}>
                {config.ffn === 'ffn_gelu' || config.ffn === 'ffn_geglu' ? 'GELU' : 'SiLU'}
              </text>

              <line x1="288" y1="488" x2="308" y2="488" stroke={borderStrong} strokeWidth="1.2" />

              <rect x="308" y="468" width="102" height="40" rx="4" fill={surfaceSub} stroke={border} />
              <text x="359" y="485" textAnchor="middle" fontSize="9" fontWeight="700" fill={ink}>
                W_down
              </text>
              <text x="359" y="499" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
                {isLooped ? `Loop ×${dims.loopPasses} passes` : `${dims.dFf} → ${dims.dModel}`}
              </text>
            </g>
          )}
        </g>

        {/* Post-FFN Norm (Peri-Norm or Post-Norm) */}
        {(isPeriNorm || isPostNorm) ? (
          <g className="cursor-pointer" onClick={() => onSelectStage?.('norm')}>
            <line x1="240" y1={isPeriNorm ? 522 : 512} x2="240" y2="527" stroke={borderStrong} strokeWidth="1.3" />
            <rect x="160" y="527" width="160" height="18" rx="4" fill={surface} stroke={cNorm} strokeWidth="1.2" />
            <text x="240" y="539.5" textAnchor="middle" fontSize="8.8" fontWeight="700" fill={cNorm}>
              Post-FFN {normShort}
            </text>
            <line x1="240" y1="545" x2="240" y2="550" stroke={borderStrong} strokeWidth="1.3" />
          </g>
        ) : (
          <line x1="240" y1="522" x2="240" y2="550" stroke={borderStrong} strokeWidth="1.3" />
        )}

        {/* Residual Addition Node #2 */}
        <circle cx="240" cy="560" r="10" fill={surface} stroke={cNorm} strokeWidth="1.8" />
        <text x="240" y="564" textAnchor="middle" fontSize="13" fontWeight="700" fill={cNorm}>+</text>
        <line x1="240" y1="570" x2="240" y2="584" stroke={borderStrong} strokeWidth="1.4" />

        {/* Looped Depth Feedback Arc */}
        {isLooped && (
          <g>
            <path
              d="M 426 560 L 444 560 L 444 190 L 428 190"
              fill="none"
              stroke={isGatedLoopNorm ? cPos : cFfn}
              strokeWidth="1.7"
              strokeDasharray="4 3"
              className={animatePulse ? 'animate-tensor-flow' : ''}
            />
            <text x="441" y="385" fontSize="7.8" fontWeight="700" fill={isGatedLoopNorm ? cPos : cFfn} transform="rotate(-90 441 385)" textAnchor="middle">
              {isGatedLoopNorm
                ? `GATED PROGRESSIVE β_k · LOOP ×${dims.loopPasses}`
                : `RECURRENT DEPTH LOOP ×${dims.loopPasses}`}
            </text>
          </g>
        )}
      </g>

      {/* Spine Connector: Block -> Final Norm */}
      <line
        x1="240"
        y1="584"
        x2="240"
        y2="606"
        stroke={borderStrong}
        strokeWidth="1.5"
        markerEnd="url(#arrowInk)"
        className={animatePulse ? 'animate-tensor-flow' : ''}
      />

      {/* ==================== STAGE 7: FINAL NORM + LM HEAD + MTP ==================== */}
      <g
        className="cursor-pointer"
        onClick={() => onSelectStage?.('output')}
      >
        <rect x="155" y="608" width="170" height="22" rx="4" fill={surface} stroke={border} />
        <text x="240" y="622.5" textAnchor="middle" fontSize="9.5" fontWeight="700" fill={inkSec}>
          Final {normShort} · [B, T, {dims.dModel}]
        </text>

        <line x1="240" y1="630" x2="240" y2="646" stroke={borderStrong} strokeWidth="1.4" markerEnd="url(#arrowInk)" />

        {/* Main LM Head */}
        <rect
          x={hasMtp ? 54 : 90}
          y="648"
          width={hasMtp ? 236 : 300}
          height="52"
          rx="6"
          fill={surface}
          stroke={highlightedStage === 'output' ? cOut : cOut}
          strokeWidth={highlightedStage === 'output' ? 2.2 : 1.4}
        />
        <rect x={hasMtp ? 54 : 90} y="648" width="4" height="52" rx="2" fill={cOut} />
        <text x={hasMtp ? 68 : 104} y="667" fontSize="11" fontWeight="700" fill={ink}>
          {out?.shortName} · Linear LM Head
        </text>
        <text x={hasMtp ? 68 : 104} y="683" fontSize="9" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
          {dims.dModel} → {physics.vocabSize.toLocaleString()} Logits{hasSoftCap ? ' · tanh(·/30)' : ''}
        </text>
        <text x={hasMtp ? 68 : 104} y="695" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={cOut}>
          {isTied ? 'WEIGHT-TIED (0 EXTRA PARAMS)' : `HEAD PARAMS: ${formatParams(physics.params.outputHead)}`}
        </text>

        {/* Auxiliary MTP-k Branch */}
        {hasMtp && (
          <g>
            <path d="M 240 638 L 368 638 L 368 648" fill="none" stroke={cOut} strokeWidth="1.4" strokeDasharray="3 2" />
            <rect x="302" y="648" width="132" height="52" rx="6" fill={isPaper ? 'rgba(225,29,72,0.07)' : 'rgba(251,113,133,0.12)'} stroke={cOut} strokeWidth="1.3" />
            <text x="368" y="667" textAnchor="middle" fontSize="9.8" fontWeight="700" fill={cOut}>
              MTP-k Heads
            </text>
            <text x="368" y="681" textAnchor="middle" fontSize="8" fontFamily="'JetBrains Mono', monospace" fill={inkSec}>
              Predict t+2, t+3…
            </text>
            <text x="368" y="694" textAnchor="middle" fontSize="7.8" fontWeight="700" fill={cFfn}>
              ~1.65× Spec Decode
            </text>
          </g>
        )}

        {/* Weight-Tied Embedding Dashed Link back to Stage 1 */}
        {isTied && (
          <g>
            <path
              d="M 90 674 L 14 674 L 14 63 L 86 63"
              fill="none"
              stroke={cOut}
              strokeWidth="1.3"
              strokeDasharray="4 3"
              markerEnd="url(#arrowOut)"
            />
            <text x="11" y="385" fontSize="8" fontWeight="700" fill={cOut} transform="rotate(-90 11 385)" textAnchor="middle">
              WEIGHT-TIED EMBEDDING (Eᵀ)
            </text>
          </g>
        )}
      </g>

      {/* Bottom Output Tensor Badge */}
      <text x="240" y="724" textAnchor="middle" fontSize="9" fontFamily="'JetBrains Mono', monospace" fill={inkMuted}>
        NEXT-TOKEN PROBABILITY DISTRIBUTION · P(x_(t+1) | x_(1..t))
      </text>
    </svg>
  );

  return (
    <>
      <div
        className="rounded-lg border flex flex-col overflow-hidden"
        style={{
          backgroundColor: 'var(--bg-panel)',
          borderColor: 'var(--border-hairline)',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        {/* Diagram Header Controls */}
        <div
          className="px-3 py-2 border-b flex items-center justify-between gap-2"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <div className="flex items-center gap-2 min-w-0">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'var(--stage-attention)' }}
            />
            <span className="font-display font-bold text-[11.5px] uppercase tracking-[0.07em] truncate">
              Live Architectural Schematic
            </span>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => setAnimatePulse((p) => !p)}
              className="p-1 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              title={animatePulse ? 'Pause forward-pass pulse' : 'Play forward-pass pulse'}
              style={{ color: 'var(--ink-secondary)' }}
            >
              {animatePulse ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            </button>
            <button
              type="button"
              onClick={exportSvg}
              className="flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-mono-num hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-secondary)' }}
              title="Download vector SVG schematic"
            >
              <Download className="w-3 h-3" />
              SVG
            </button>
            <button
              type="button"
              onClick={exportPng}
              className="flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-mono-num hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-secondary)' }}
              title="Download high-res PNG schematic"
            >
              <ImageIcon className="w-3 h-3" />
              PNG
            </button>
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="p-1 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              title="Expand full-screen blueprint"
              style={{ color: 'var(--ink-secondary)' }}
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* SVG Viewport */}
        <div className="p-2 flex items-center justify-center">
          {renderSvg('w-full h-auto max-h-[650px]')}
        </div>
      </div>

      {/* Fullscreen Lightbox Blueprint Modal */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs"
          onClick={() => setExpanded(false)}
        >
          <div
            className="rounded-xl border max-w-2xl w-full max-h-[94vh] flex flex-col overflow-hidden"
            style={{
              backgroundColor: 'var(--bg-panel)',
              borderColor: 'var(--border-strong)',
              boxShadow: 'var(--shadow-float)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              className="px-4 py-2.5 border-b flex items-center justify-between"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              <div>
                <div className="font-display font-bold text-[13px]">{config.name}</div>
                <div className="text-[11px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                  {physics.decoderTypeLabel} · {physics.layerMixSummary} · {formatKiB(physics.kvKiBPerTokenBf16)}/tok BF16
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={exportSvg}
                  className="px-2.5 py-1 rounded border text-[11px] font-mono-num cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
                  style={{ borderColor: 'var(--border-hairline)' }}
                >
                  Export SVG
                </button>
                <button
                  type="button"
                  onClick={exportPng}
                  className="px-2.5 py-1 rounded border text-[11px] font-mono-num cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
                  style={{ borderColor: 'var(--border-hairline)' }}
                >
                  Export PNG
                </button>
                <button
                  type="button"
                  onClick={() => setExpanded(false)}
                  className="px-2.5 py-1 rounded text-[11px] font-semibold cursor-pointer"
                  style={{ backgroundColor: 'var(--ink-primary)', color: 'var(--ink-inverse)' }}
                >
                  Close
                </button>
              </div>
            </div>
            <div className="p-4 overflow-y-auto flex justify-center blueprint-grid">
              {renderSvg('w-full max-w-[560px] h-auto')}
            </div>
          </div>
        </div>
      )}
    </>
  );
};
