import React from 'react';
import { GripVertical } from 'lucide-react';
import { STAGE_MAP } from '../data/components';
import type { ComponentDef } from '../types/architecture';

interface DragOverlayProps {
  comp: ComponentDef | null;
  x: number;
  y: number;
  isOverValidTarget: boolean;
}

export const DragOverlay: React.FC<DragOverlayProps> = ({
  comp,
  x,
  y,
  isOverValidTarget,
}) => {
  if (!comp) return null;
  const st = STAGE_MAP[comp.stage];

  return (
    <div
      className="fixed pointer-events-none z-[100] w-[270px] rounded-lg border-2 p-2.5 transition-transform duration-75"
      style={{
        left: x + 14,
        top: y - 24,
        backgroundColor: 'var(--bg-panel-elevated)',
        borderColor: st.cssVar,
        boxShadow: 'var(--shadow-float)',
        transform: isOverValidTarget ? 'scale(1.04) rotate(0deg)' : 'scale(1) rotate(-1.5deg)',
      }}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <GripVertical className="w-3.5 h-3.5 shrink-0" style={{ color: st.cssVar }} />
          <span className="text-[12.5px] font-bold truncate">{comp.name}</span>
        </div>
        <span
          className="text-[9.5px] font-mono-num font-bold px-1.5 py-0.5 rounded shrink-0"
          style={{ backgroundColor: st.softVar, color: st.cssVar }}
        >
          {comp.badge}
        </span>
      </div>
      <div
        className="mt-1 text-[10.5px] font-mono-num flex items-center justify-between"
        style={{ color: isOverValidTarget ? st.cssVar : 'var(--ink-muted)' }}
      >
        <span>Stage 0{st.order}: {st.shortLabel}</span>
        <span className="font-semibold">
          {isOverValidTarget ? '● Release to Snap' : 'Drag to Slot →'}
        </span>
      </div>
    </div>
  );
};
