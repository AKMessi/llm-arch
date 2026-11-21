import React, { useMemo } from 'react';
import {
  ArrowLeft,
  Camera,
  Check,
  GitCompare,
  Sparkles,
} from 'lucide-react';
import { COMPONENT_MAP, STAGES } from '../data/components';
import { galleryModelToConfig, PRESET_BLUEPRINTS } from '../data/presets';
import {
  computePhysics,
  formatContext,
  formatGb,
  formatKiB,
  formatParams,
} from '../engine/physics';
import type {
  ArchitectureConfig,
  ComponentDef,
  GalleryModel,
  PhysicsReport,
} from '../types/architecture';

interface CompareViewProps {
  configA: ArchitectureConfig;
  physicsA: PhysicsReport;
  configB: ArchitectureConfig;
  onChangeConfigB: (cfg: ArchitectureConfig) => void;
  onSnapshotAToB: () => void;
  onCherryPickFromB: (comp: ComponentDef) => void;
  onReplaceAWithB: () => void;
  galleryModels: GalleryModel[];
}

export const CompareView: React.FC<CompareViewProps> = ({
  configA,
  physicsA,
  configB,
  onChangeConfigB,
  onSnapshotAToB,
  onCherryPickFromB,
  onReplaceAWithB,
  galleryModels,
}) => {
  const physicsB = useMemo(() => computePhysics(configB), [configB]);

  const metrics = [
    {
      label: 'Total Parameters',
      aVal: formatParams(physicsA.params.total),
      bVal: formatParams(physicsB.params.total),
      aNum: physicsA.params.total,
      bNum: physicsB.params.total,
      lowerIsBetter: false,
    },
    {
      label: 'Active Params / Token',
      aVal: `${formatParams(physicsA.params.active)} (${physicsA.params.activeRatioPct.toFixed(1)}%)`,
      bVal: `${formatParams(physicsB.params.active)} (${physicsB.params.activeRatioPct.toFixed(1)}%)`,
      aNum: physicsA.params.active,
      bNum: physicsB.params.active,
      lowerIsBetter: true,
    },
    {
      label: 'KV-Cache / Token (BF16)',
      aVal: formatKiB(physicsA.kvKiBPerTokenBf16),
      bVal: formatKiB(physicsB.kvKiBPerTokenBf16),
      aNum: physicsA.kvKiBPerTokenBf16,
      bNum: physicsB.kvKiBPerTokenBf16,
      lowerIsBetter: true,
    },
    {
      label: 'KV-Cache @ 128K Context',
      aVal: formatGb(physicsA.kvCurve[3].currentGb),
      bVal: formatGb(physicsB.kvCurve[3].currentGb),
      aNum: physicsA.kvCurve[3].currentGb,
      bNum: physicsB.kvCurve[3].currentGb,
      lowerIsBetter: true,
    },
    {
      label: 'Total VRAM (Target Ctx)',
      aVal: formatGb(physicsA.totalVramGb),
      bVal: formatGb(physicsB.totalVramGb),
      aNum: physicsA.totalVramGb,
      bNum: physicsB.totalVramGb,
      lowerIsBetter: true,
    },
    {
      label: 'Est. H100 Decode Speed',
      aVal: `${Math.round(physicsA.estDecodeTokSecH100)} tok/s`,
      bVal: `${Math.round(physicsB.estDecodeTokSecH100)} tok/s`,
      aNum: physicsA.estDecodeTokSecH100,
      bNum: physicsB.estDecodeTokSecH100,
      lowerIsBetter: false,
    },
    {
      label: 'Context Window',
      aVal: formatContext(configA.dims.contextLength),
      bVal: formatContext(configB.dims.contextLength),
      aNum: configA.dims.contextLength,
      bNum: configB.dims.contextLength,
      lowerIsBetter: false,
    },
  ];

  return (
    <main className="flex-1 h-full overflow-y-auto blueprint-grid p-4">
      <div className="max-w-[1120px] mx-auto space-y-4">
        {/* Top Selector Banner */}
        <section
          className="rounded-lg border p-4 flex flex-wrap items-center justify-between gap-3"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="flex items-center gap-2.5">
            <GitCompare className="w-4.5 h-4.5" style={{ color: 'var(--stage-attention)' }} />
            <div>
              <h2 className="font-display font-bold text-[15px]">
                Architecture Diff &amp; Cherry-Pick Arena
              </h2>
              <p className="text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
                Benchmark your active Blueprint A against any real-world reference model — and pull individual blocks straight into your stack.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onSnapshotAToB}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-[11.5px] font-medium hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              <Camera className="w-3.5 h-3.5" />
              Snapshot A → B
            </button>

            <select
              onChange={(e) => {
                const val = e.target.value;
                const preset = PRESET_BLUEPRINTS.find((p) => p.id === val);
                if (preset) {
                  onChangeConfigB({ ...preset.config, dims: { ...preset.config.dims } });
                  return;
                }
                const gModel = galleryModels.find((m) => m.id === val);
                if (gModel) onChangeConfigB(galleryModelToConfig(gModel));
              }}
              value=""
              className="px-2.5 py-1.5 rounded-md border text-[11.5px] font-medium cursor-pointer focus:outline-none max-w-[230px]"
              style={{
                backgroundColor: 'var(--bg-slot)',
                borderColor: 'var(--border-strong)',
                color: 'var(--ink-primary)',
              }}
            >
              <option value="" disabled>
                Load Reference into Slot B…
              </option>
              <optgroup label="12 Iconic Blueprints">
                {PRESET_BLUEPRINTS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.kvBadge})
                  </option>
                ))}
              </optgroup>
              <optgroup label="All 120 Gallery Models">
                {galleryModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} — {m.company}
                  </option>
                ))}
              </optgroup>
            </select>

            <button
              type="button"
              onClick={onReplaceAWithB}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11.5px] font-semibold cursor-pointer"
              style={{
                backgroundColor: 'var(--ink-primary)',
                color: 'var(--ink-inverse)',
              }}
            >
              <Sparkles className="w-3.5 h-3.5" />
              Load B onto Canvas
            </button>
          </div>
        </section>

        {/* Quantitative Benchmark Bars */}
        <section
          className="rounded-lg border p-4 space-y-3"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="grid grid-cols-12 items-center pb-2 border-b text-[11px] font-bold uppercase tracking-wider" style={{ borderColor: 'var(--border-hairline)' }}>
            <div className="col-span-4" style={{ color: 'var(--ink-muted)' }}>
              Metric
            </div>
            <div className="col-span-4 flex items-center gap-1.5" style={{ color: 'var(--stage-attention)' }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--stage-attention)' }} />
              A: {configA.name}
            </div>
            <div className="col-span-4 flex items-center gap-1.5" style={{ color: 'var(--stage-tokenizer)' }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--stage-tokenizer)' }} />
              B: {configB.name}
            </div>
          </div>

          <div className="space-y-3">
            {metrics.map((m) => {
              const maxVal = Math.max(0.001, m.aNum, m.bNum);
              const pctA = Math.max(4, (m.aNum / maxVal) * 100);
              const pctB = Math.max(4, (m.bNum / maxVal) * 100);
              const aWins =
                m.aNum !== m.bNum && (m.lowerIsBetter ? m.aNum < m.bNum : m.aNum > m.bNum);
              const bWins =
                m.aNum !== m.bNum && (m.lowerIsBetter ? m.bNum < m.aNum : m.bNum > m.aNum);

              return (
                <div key={m.label} className="grid grid-cols-12 items-center gap-3">
                  <div className="col-span-4 text-[12px] font-medium" style={{ color: 'var(--ink-secondary)' }}>
                    {m.label}
                  </div>
                  <div className="col-span-4 space-y-1">
                    <div className="flex items-center justify-between text-[11.5px] font-mono-num">
                      <span className={aWins ? 'font-bold' : ''}>{m.aVal}</span>
                      {aWins && (
                        <span className="text-[9.5px] font-bold px-1.5 py-0.2 rounded" style={{ backgroundColor: 'var(--stage-ffn-soft)', color: 'var(--stage-ffn)' }}>
                          BETTER
                        </span>
                      )}
                    </div>
                    <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-panel-subtle)' }}>
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${pctA}%`, backgroundColor: 'var(--stage-attention)' }}
                      />
                    </div>
                  </div>
                  <div className="col-span-4 space-y-1">
                    <div className="flex items-center justify-between text-[11.5px] font-mono-num">
                      <span className={bWins ? 'font-bold' : ''}>{m.bVal}</span>
                      {bWins && (
                        <span className="text-[9.5px] font-bold px-1.5 py-0.2 rounded" style={{ backgroundColor: 'var(--stage-ffn-soft)', color: 'var(--stage-ffn)' }}>
                          BETTER
                        </span>
                      )}
                    </div>
                    <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-panel-subtle)' }}>
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${pctB}%`, backgroundColor: 'var(--stage-tokenizer)' }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Stage-by-Stage Structural Diff + Cherry-Pick */}
        <section
          className="rounded-lg border p-4 space-y-2.5"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="flex items-center justify-between pb-1">
            <h3 className="font-display font-bold text-[12px] uppercase tracking-[0.07em]">
              7-Stage Architectural Diff (Click to Cherry-Pick from B → A)
            </h3>
            <span className="text-[11px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
              {STAGES.filter((s) => configA[s.id] === configB[s.id]).length} / 7 identical stages
            </span>
          </div>

          <div className="space-y-1.5">
            {STAGES.map((stage) => {
              const compA = COMPONENT_MAP[configA[stage.id]];
              const compB = COMPONENT_MAP[configB[stage.id]];
              const same = compA.id === compB.id;
              return (
                <div
                  key={stage.id}
                  className="grid grid-cols-12 items-center gap-2 p-2.5 rounded-md border"
                  style={{
                    backgroundColor: same ? 'var(--bg-slot)' : stage.softVar,
                    borderColor: same ? 'var(--border-hairline)' : stage.cssVar,
                  }}
                >
                  <div className="col-span-3 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: stage.cssVar }} />
                    <span className="text-[11.5px] font-bold truncate">{stage.shortLabel}</span>
                  </div>
                  <div className="col-span-4">
                    <div className="text-[12px] font-semibold truncate">{compA.name}</div>
                    <div className="text-[10px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                      {compA.badge}
                    </div>
                  </div>
                  <div className="col-span-1 flex justify-center">
                    {same ? (
                      <Check className="w-4 h-4" style={{ color: 'var(--stage-ffn)' }} />
                    ) : (
                      <button
                        type="button"
                        onClick={() => onCherryPickFromB(compB)}
                        title={`Pull ${compB.name} into Blueprint A`}
                        className="p-1.5 rounded-full border cursor-pointer hover:scale-110 transition-transform"
                        style={{
                          backgroundColor: 'var(--bg-panel)',
                          borderColor: stage.cssVar,
                          color: stage.cssVar,
                        }}
                      >
                        <ArrowLeft className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                  <div className="col-span-4">
                    <div className="text-[12px] font-semibold truncate">{compB.name}</div>
                    <div className="text-[10px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                      {compB.badge}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </main>
  );
};
