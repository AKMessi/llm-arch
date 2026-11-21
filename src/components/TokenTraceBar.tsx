import React, { useEffect } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  Radio,
  RotateCcw,
  X,
} from 'lucide-react';
import { STAGE_MAP } from '../data/components';
import type { ForwardTraceStep, StageId } from '../types/architecture';

interface TokenTraceBarProps {
  steps: ForwardTraceStep[];
  activeStepIdx: number | null;
  isPlaying: boolean;
  onStart: () => void;
  onStop: () => void;
  onTogglePlay: () => void;
  onStepChange: (idx: number) => void;
  onHighlightStage?: (stage: StageId | null) => void;
}

export const TokenTraceBar: React.FC<TokenTraceBarProps> = ({
  steps,
  activeStepIdx,
  isPlaying,
  onStart,
  onStop,
  onTogglePlay,
  onStepChange,
  onHighlightStage,
}) => {
  useEffect(() => {
    if (activeStepIdx === null) {
      onHighlightStage?.(null);
      return;
    }
    const step = steps[activeStepIdx];
    if (step) onHighlightStage?.(step.stage);
  }, [activeStepIdx, steps, onHighlightStage]);

  useEffect(() => {
    if (!isPlaying || activeStepIdx === null) return;
    const timer = window.setInterval(() => {
      onStepChange((activeStepIdx + 1) % steps.length);
    }, 2400);
    return () => window.clearInterval(timer);
  }, [isPlaying, activeStepIdx, steps.length, onStepChange]);

  const totalUs = steps.reduce((sum, s) => sum + s.latencyUs, 0) || 1;

  if (activeStepIdx === null) {
    return (
      <div
        className="rounded-lg border px-3 py-2 flex flex-wrap items-center justify-between gap-2"
        style={{
          backgroundColor: 'var(--bg-panel)',
          borderColor: 'var(--border-hairline)',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        <div className="flex items-center gap-2 text-[11.5px]">
          <Radio className="w-3.5 h-3.5" style={{ color: 'var(--stage-attention)' }} />
          <span className="font-semibold">Live Forward-Pass Token Tracer</span>
          <span className="hidden sm:inline" style={{ color: 'var(--ink-muted)' }}>
            — Step a single token through your exact architecture to inspect tensor shapes, memory reads &amp; latency.
          </span>
        </div>
        <button
          type="button"
          onClick={onStart}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold cursor-pointer transition-opacity hover:opacity-90"
          style={{
            backgroundColor: 'var(--stage-attention)',
            color: '#fff',
          }}
        >
          <Play className="w-3 h-3 fill-current" />
          Trace 1 Token
        </button>
      </div>
    );
  }

  const cur = steps[activeStepIdx];
  const st = STAGE_MAP[cur.stage];
  const pctTime = Math.round((cur.latencyUs / totalUs) * 100);

  return (
    <div
      className="rounded-lg border p-3 space-y-2.5"
      style={{
        backgroundColor: 'var(--bg-panel)',
        borderColor: st.cssVar,
        boxShadow: 'var(--shadow-card)',
      }}
    >
      {/* Top Bar: Step Title + Transport Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="text-[10px] font-mono-num font-bold px-2 py-0.5 rounded uppercase shrink-0"
            style={{ backgroundColor: st.cssVar, color: '#fff' }}
          >
            STEP {cur.step} / {steps.length}
          </span>
          <span className="font-display font-bold text-[13px] truncate">
            {cur.title}
          </span>
          <span
            className="text-[10px] font-mono-num px-1.5 py-0.5 rounded border hidden md:inline"
            style={{
              backgroundColor: 'var(--bg-slot)',
              borderColor: 'var(--border-hairline)',
              color: st.cssVar,
            }}
          >
            {cur.highlightFact}
          </span>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={() => onStepChange((activeStepIdx - 1 + steps.length) % steps.length)}
            className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
            style={{ borderColor: 'var(--border-hairline)' }}
            title="Previous step"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onTogglePlay}
            className="flex items-center gap-1 px-2 py-1 rounded border text-[10.5px] font-semibold cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
            style={{ borderColor: 'var(--border-hairline)' }}
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => onStepChange((activeStepIdx + 1) % steps.length)}
            className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
            style={{ borderColor: 'var(--border-hairline)' }}
            title="Next step"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => onStepChange(0)}
            className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
            style={{ borderColor: 'var(--border-hairline)' }}
            title="Restart from Step 1"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onStop}
            className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
            style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-muted)' }}
            title="Close Tracer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Stepper Timeline Pills */}
      <div className="grid grid-cols-7 gap-1">
        {steps.map((s, idx) => {
          const sm = STAGE_MAP[s.stage];
          const isCur = idx === activeStepIdx;
          const isPast = idx < activeStepIdx;
          return (
            <button
              key={s.step}
              type="button"
              onClick={() => onStepChange(idx)}
              className="text-left p-1.5 rounded border transition-all cursor-pointer"
              style={{
                backgroundColor: isCur ? sm.softVar : 'var(--bg-slot)',
                borderColor: isCur ? sm.cssVar : isPast ? 'var(--border-strong)' : 'var(--border-hairline)',
              }}
            >
              <div className="flex items-center justify-between text-[9px] font-mono-num">
                <span style={{ color: isCur ? sm.cssVar : 'var(--ink-muted)', fontWeight: isCur ? 700 : 500 }}>
                  0{s.step}
                </span>
                <span style={{ color: 'var(--ink-muted)' }}>
                  {Math.round((s.latencyUs / totalUs) * 100)}%
                </span>
              </div>
              <div className="text-[10px] font-semibold truncate mt-0.5">
                {sm.shortLabel}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Step Technical Readout */}
      <div
        className="rounded-md border p-2.5 grid grid-cols-1 md:grid-cols-12 gap-2.5 items-center text-[11px]"
        style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
      >
        <div className="md:col-span-7 space-y-1">
          <div className="font-medium leading-snug">{cur.opSummary}</div>
          <code className="block text-[10px] font-mono-num truncate" style={{ color: st.cssVar }}>
            {cur.mathNote}
          </code>
        </div>

        <div className="md:col-span-5 grid grid-cols-3 gap-1.5 font-mono-num text-[10px]">
          <div className="p-1.5 rounded border" style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}>
            <div style={{ color: 'var(--ink-muted)' }}>TENSOR OUT</div>
            <div className="font-bold truncate mt-0.5">{cur.tensorOut}</div>
          </div>
          <div className="p-1.5 rounded border" style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}>
            <div style={{ color: 'var(--ink-muted)' }}>MEM READ</div>
            <div className="font-bold mt-0.5">
              {cur.memoryReadMb >= 1024
                ? `${(cur.memoryReadMb / 1024).toFixed(1)} GB`
                : `${cur.memoryReadMb} MB`}
            </div>
          </div>
          <div className="p-1.5 rounded border" style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}>
            <div style={{ color: 'var(--ink-muted)' }}>STAGE TIME</div>
            <div className="font-bold mt-0.5" style={{ color: st.cssVar }}>
              {cur.latencyUs >= 1000
                ? `${(cur.latencyUs / 1000).toFixed(2)}ms`
                : `${cur.latencyUs}µs`}{' '}
              ({pctTime}%)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
