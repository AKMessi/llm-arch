import React, { useState } from 'react';
import {
  Award,
  CheckCircle2,
  Circle,
  Lightbulb,
  Pin,
  Play,
  Target,
  Trophy,
} from 'lucide-react';
import type { EvaluatedChallenge } from '../data/challenges';

interface MissionsViewProps {
  evaluated: EvaluatedChallenge[];
  solvedIds: Set<string>;
  pinnedMissionId: string | null;
  onPinMission: (id: string | null) => void;
  onStartMission: (missionId: string, starterPresetId: string) => void;
}

export const MissionsView: React.FC<MissionsViewProps> = ({
  evaluated,
  solvedIds,
  pinnedMissionId,
  onPinMission,
  onStartMission,
}) => {
  const [revealedHints, setRevealedHints] = useState<Set<string>>(new Set());

  const solvedCount = solvedIds.size;
  const totalCount = evaluated.length;

  const rank =
    solvedCount >= 8
      ? 'Frontier Architect'
      : solvedCount >= 5
        ? 'Principal ML Systems Engineer'
        : solvedCount >= 3
          ? 'Staff Transformer Designer'
          : solvedCount >= 1
            ? 'Decoder Apprentice'
            : 'Drafting Recruit';

  const toggleHint = (id: string) => {
    setRevealedHints((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <main className="flex-1 h-full overflow-y-auto blueprint-grid p-4">
      <div className="max-w-[1120px] mx-auto space-y-4">
        {/* Top Progress Banner */}
        <section
          className="rounded-lg border p-4 flex flex-wrap items-center justify-between gap-4"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Trophy className="w-4.5 h-4.5" style={{ color: 'var(--stage-hybrid)' }} />
              <h2 className="font-display font-bold text-[16px]">
                LLM Systems Design Missions
              </h2>
              <span
                className="text-[10.5px] font-mono-num font-bold px-2 py-0.5 rounded"
                style={{
                  backgroundColor: 'var(--stage-ffn-soft)',
                  color: 'var(--stage-ffn)',
                }}
              >
                {rank.toUpperCase()}
              </span>
            </div>
            <p className="text-[12px]" style={{ color: 'var(--ink-secondary)' }}>
              Real-world hardware &amp; scaling constraints evaluated live against your blueprint. Pin any mission to track its checklist live on the Blueprint Canvas while you build!
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right font-mono-num">
              <div className="text-[20px] font-bold leading-none">
                {solvedCount} <span className="text-[13px] font-normal" style={{ color: 'var(--ink-muted)' }}>/ {totalCount}</span>
              </div>
              <div className="text-[10px] mt-1" style={{ color: 'var(--ink-muted)' }}>
                MISSIONS CLEARED
              </div>
            </div>
            <div className="w-28 h-2.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-panel-subtle)' }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${(solvedCount / Math.max(1, totalCount)) * 100}%`,
                  backgroundColor: 'var(--stage-ffn)',
                }}
              />
            </div>
          </div>
        </section>

        {/* Missions Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5">
          {evaluated.map(({ challenge, constraints, passedCount, totalCount: cTot, completed }) => {
            const everSolved = solvedIds.has(challenge.id) || completed;
            const isPinned = pinnedMissionId === challenge.id;
            const hintOpen = revealedHints.has(challenge.id);
            const diffColor =
              challenge.difficulty === 'Starter'
                ? 'var(--stage-ffn)'
                : challenge.difficulty === 'Engineer'
                  ? 'var(--stage-tokenizer)'
                  : challenge.difficulty === 'Principal'
                    ? 'var(--stage-attention)'
                    : 'var(--stage-output)';

            return (
              <article
                key={challenge.id}
                className="rounded-lg border p-4 flex flex-col justify-between gap-3 transition-all"
                style={{
                  backgroundColor: 'var(--bg-panel)',
                  borderColor: completed
                    ? 'var(--stage-ffn)'
                    : isPinned
                      ? 'var(--stage-attention)'
                      : 'var(--border-hairline)',
                  boxShadow: 'var(--shadow-card)',
                }}
              >
                <div className="space-y-2.5">
                  {/* Card Top Header */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="text-[10px] font-mono-num font-bold px-1.5 py-0.5 rounded"
                          style={{
                            backgroundColor: 'var(--bg-panel-subtle)',
                            color: 'var(--ink-secondary)',
                          }}
                        >
                          MISSION {challenge.number}
                        </span>
                        <span
                          className="text-[10px] font-mono-num font-semibold"
                          style={{ color: diffColor }}
                        >
                          {challenge.difficulty} · {challenge.category}
                        </span>
                      </div>
                      <h3 className="font-display font-bold text-[15px]">
                        {challenge.title}
                      </h3>
                    </div>

                    {everSolved ? (
                      <span
                        className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10.5px] font-mono-num font-bold shrink-0"
                        style={{
                          backgroundColor: 'var(--stage-ffn-soft)',
                          color: 'var(--stage-ffn)',
                        }}
                      >
                        <Award className="w-3.5 h-3.5" />
                        CLEARED
                      </span>
                    ) : (
                      <span
                        className="text-[11px] font-mono-num font-bold px-2 py-0.5 rounded shrink-0"
                        style={{
                          backgroundColor: 'var(--bg-panel-subtle)',
                          color: 'var(--ink-secondary)',
                        }}
                      >
                        {passedCount}/{cTot}
                      </span>
                    )}
                  </div>

                  <p className="text-[12px] leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
                    {challenge.brief}
                  </p>

                  {/* Live Constraint Checklist */}
                  <div
                    className="rounded-md border p-2.5 space-y-1.5"
                    style={{
                      backgroundColor: 'var(--bg-slot)',
                      borderColor: 'var(--border-hairline)',
                    }}
                  >
                    {constraints.map((c) => (
                      <div key={c.id} className="flex items-center justify-between gap-2 text-[11.5px]">
                        <span className="flex items-center gap-1.5 min-w-0">
                          {c.passed ? (
                            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color: 'var(--stage-ffn)' }} />
                          ) : (
                            <Circle className="w-3.5 h-3.5 shrink-0 opacity-35" />
                          )}
                          <span className={c.passed ? 'font-medium' : ''} style={{ color: c.passed ? 'var(--ink-primary)' : 'var(--ink-secondary)' }}>
                            {c.label}
                          </span>
                        </span>
                        <span
                          className="font-mono-num text-[10.5px] font-semibold shrink-0"
                          style={{ color: c.passed ? 'var(--stage-ffn)' : 'var(--ink-muted)' }}
                        >
                          {c.currentText}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Optional Hint Box */}
                  {hintOpen && (
                    <div
                      className="rounded-md border px-2.5 py-2 text-[11px] leading-relaxed"
                      style={{
                        backgroundColor: 'var(--stage-hybrid-soft)',
                        borderColor: 'var(--stage-hybrid)',
                        color: 'var(--ink-primary)',
                      }}
                    >
                      <strong>Architect Hint:</strong> {challenge.hint}
                    </div>
                  )}

                  <div className="text-[10.5px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                    {challenge.realWorldRef}
                  </div>
                </div>

                {/* Action Footer */}
                <div
                  className="pt-2.5 border-t flex items-center justify-between gap-2"
                  style={{ borderColor: 'var(--border-hairline)' }}
                >
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => onStartMission(challenge.id, challenge.starterPresetId)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11.5px] font-semibold cursor-pointer"
                      style={{
                        backgroundColor: 'var(--ink-primary)',
                        color: 'var(--ink-inverse)',
                      }}
                    >
                      <Play className="w-3 h-3 fill-current" />
                      Launch on Canvas
                    </button>

                    <button
                      type="button"
                      onClick={() => onPinMission(isPinned ? null : challenge.id)}
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-md border text-[11px] font-medium cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
                      style={{
                        borderColor: isPinned ? 'var(--stage-attention)' : 'var(--border-hairline)',
                        color: isPinned ? 'var(--stage-attention)' : 'var(--ink-secondary)',
                      }}
                    >
                      <Pin className="w-3 h-3" />
                      {isPinned ? 'Pinned' : 'Pin HUD'}
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => toggleHint(challenge.id)}
                    className="flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
                    style={{ color: 'var(--ink-muted)' }}
                  >
                    <Lightbulb className="w-3.5 h-3.5" />
                    {hintOpen ? 'Hide Hint' : 'Hint'}
                  </button>
                </div>
              </article>
            );
          })}
        </div>

        {/* Bottom Note */}
        <div
          className="rounded-lg border p-3 flex items-center gap-2 text-[11.5px]"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            color: 'var(--ink-secondary)',
          }}
        >
          <Target className="w-4 h-4 shrink-0" style={{ color: 'var(--stage-tokenizer)' }} />
          <span>
            Missions evaluate dynamically against your active canvas in real time — even custom architectures built outside the starter preset count toward clearing challenges!
          </span>
        </div>
      </div>
    </main>
  );
};
