import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bookmark,
  Blocks,
  Image as ImageIcon,
  Layers,
  Moon,
  Search,
  Shuffle,
  Trophy,
  X,
} from 'lucide-react';
import { ARCHITECTURE_CHALLENGES } from '../data/challenges';
import { COMPONENTS, STAGE_MAP } from '../data/components';
import type { ComponentDef, GalleryModel } from '../types/architecture';

interface CommandItem {
  id: string;
  group: 'Action' | 'Architectural Block' | 'Gallery Model' | 'Mission';
  title: string;
  subtitle: string;
  badge?: string;
  accent?: string;
  onSelect: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  galleryModels: GalleryModel[];
  onEquipComponent: (comp: ComponentDef) => void;
  onCloneModel: (model: GalleryModel) => void;
  onStartMission: (missionId: string, starterPresetId: string) => void;
  onExportSpecCard: () => void;
  onSaveBlueprint: () => void;
  onRandomize: () => void;
  onToggleTheme: () => void;
  onOpenGallery: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  open,
  onClose,
  galleryModels,
  onEquipComponent,
  onCloneModel,
  onStartMission,
  onExportSpecCard,
  onSaveBlueprint,
  onRandomize,
  onToggleTheme,
  onOpenGallery,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setQuery('');
      setSelectedIdx(0);
      setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [open]);

  const items = useMemo<CommandItem[]>(() => {
    const actions: CommandItem[] = [
      {
        id: 'act-spec-card',
        group: 'Action',
        title: 'Export Viral 1200×675 Spec Card PNG',
        subtitle: 'Generate a shareable Swiss poster card of your custom architecture',
        badge: 'PNG',
        onSelect: onExportSpecCard,
      },
      {
        id: 'act-save',
        group: 'Action',
        title: 'Save Current Blueprint to Local Vault',
        subtitle: 'Bookmark this architecture to compare or reload later',
        badge: 'VAULT',
        onSelect: onSaveBlueprint,
      },
      {
        id: 'act-remix',
        group: 'Action',
        title: 'Synthesize Random Experimental Architecture',
        subtitle: 'Mix attention, hybrid, tokenizer & MoE blocks randomly',
        badge: 'REMIX',
        onSelect: onRandomize,
      },
      {
        id: 'act-gallery',
        group: 'Action',
        title: `Open ${galleryModels.length}-Model Architecture Gallery`,
        subtitle: `Browse & filter all ${galleryModels.length} real open-weight models`,
        badge: String(galleryModels.length),
        onSelect: onOpenGallery,
      },
      {
        id: 'act-theme',
        group: 'Action',
        title: 'Toggle Paper / Obsidian Blueprint Theme',
        subtitle: 'Switch between archival drafting paper and dark obsidian mode',
        badge: 'THEME',
        onSelect: onToggleTheme,
      },
    ];

    const blocks: CommandItem[] = COMPONENTS.map((c) => ({
      id: `blk-${c.id}`,
      group: 'Architectural Block',
      title: c.name,
      subtitle: `${STAGE_MAP[c.stage].label} · ${c.tagline}`,
      badge: c.badge,
      accent: STAGE_MAP[c.stage].cssVar,
      onSelect: () => onEquipComponent(c),
    }));

    const missions: CommandItem[] = ARCHITECTURE_CHALLENGES.map((m) => ({
      id: `mis-${m.id}`,
      group: 'Mission',
      title: `Mission ${m.number}: ${m.title}`,
      subtitle: m.brief,
      badge: m.difficulty,
      accent: 'var(--stage-hybrid)',
      onSelect: () => onStartMission(m.id, m.starterPresetId),
    }));

    const models: CommandItem[] = galleryModels.map((m) => ({
      id: `mod-${m.id}`,
      group: 'Gallery Model',
      title: m.name,
      subtitle: `${m.company} · ${m.decoderType} · ${m.attention}`,
      badge: m.kvCachePerTokenRaw || m.scale.slice(0, 12),
      accent: 'var(--stage-tokenizer)',
      onSelect: () => onCloneModel(m),
    }));

    const all = [...actions, ...blocks, ...missions, ...models];
    const q = query.trim().toLowerCase();
    if (!q) return [...actions, ...blocks.slice(0, 8), ...missions.slice(0, 4), ...models.slice(0, 6)];

    return all
      .filter(
        (it) =>
          it.title.toLowerCase().includes(q) ||
          it.subtitle.toLowerCase().includes(q) ||
          (it.badge?.toLowerCase().includes(q) ?? false)
      )
      .slice(0, 24);
  }, [
    query,
    galleryModels,
    onEquipComponent,
    onCloneModel,
    onStartMission,
    onExportSpecCard,
    onSaveBlueprint,
    onRandomize,
    onToggleTheme,
    onOpenGallery,
  ]);

  useEffect(() => {
    setSelectedIdx(0);
  }, [query]);

  if (!open) return null;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(items.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter' && items[selectedIdx]) {
      e.preventDefault();
      items[selectedIdx].onSelect();
      onClose();
    }
  };

  const groupIcon = (g: CommandItem['group'], id: string) => {
    if (g === 'Architectural Block') return <Blocks className="w-3.5 h-3.5 shrink-0" />;
    if (g === 'Gallery Model') return <Layers className="w-3.5 h-3.5 shrink-0" />;
    if (g === 'Mission') return <Trophy className="w-3.5 h-3.5 shrink-0" />;
    if (id === 'act-spec-card') return <ImageIcon className="w-3.5 h-3.5 shrink-0" />;
    if (id === 'act-save') return <Bookmark className="w-3.5 h-3.5 shrink-0" />;
    if (id === 'act-theme') return <Moon className="w-3.5 h-3.5 shrink-0" />;
    return <Shuffle className="w-3.5 h-3.5 shrink-0" />;
  };

  return (
    <div
      className="fixed inset-0 z-[80] flex items-start justify-center pt-[10vh] p-4 bg-black/50 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className="w-full max-w-[620px] rounded-xl border overflow-hidden flex flex-col max-h-[72vh]"
        style={{
          backgroundColor: 'var(--bg-panel-elevated)',
          borderColor: 'var(--border-strong)',
          boxShadow: 'var(--shadow-float)',
        }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Search Header */}
        <div
          className="px-3.5 py-2.5 border-b flex items-center gap-2.5"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <Search className="w-4 h-4 shrink-0" style={{ color: 'var(--ink-muted)' }} />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Jump to any of ${COMPONENTS.length} blocks, ${galleryModels.length} models, ${ARCHITECTURE_CHALLENGES.length} missions, or actions…`}
            className="flex-1 bg-transparent text-[13.5px] focus:outline-none"
            style={{ color: 'var(--ink-primary)' }}
          />
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
            style={{ color: 'var(--ink-muted)' }}
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {items.length === 0 ? (
            <div className="py-10 text-center text-[12px]" style={{ color: 'var(--ink-muted)' }}>
              No blocks, models, or commands matching “{query}”
            </div>
          ) : (
            items.map((item, idx) => {
              const active = idx === selectedIdx;
              return (
                <button
                  key={item.id}
                  type="button"
                  onMouseEnter={() => setSelectedIdx(idx)}
                  onClick={() => {
                    item.onSelect();
                    onClose();
                  }}
                  className="w-full text-left px-3 py-2 rounded-lg flex items-center justify-between gap-3 cursor-pointer transition-colors"
                  style={{
                    backgroundColor: active ? 'var(--bg-panel-subtle)' : 'transparent',
                  }}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span style={{ color: item.accent ?? 'var(--ink-muted)' }}>
                      {groupIcon(item.group, item.id)}
                    </span>
                    <div className="min-w-0">
                      <div className="text-[12.5px] font-semibold truncate">{item.title}</div>
                      <div className="text-[11px] truncate" style={{ color: 'var(--ink-muted)' }}>
                        {item.subtitle}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {item.badge && (
                      <span
                        className="text-[10px] font-mono-num font-semibold px-1.5 py-0.5 rounded border"
                        style={{
                          borderColor: 'var(--border-hairline)',
                          color: item.accent ?? 'var(--ink-secondary)',
                          backgroundColor: 'var(--bg-panel)',
                        }}
                      >
                        {item.badge}
                      </span>
                    )}
                    <span className="text-[9.5px] font-mono-num uppercase hidden sm:inline" style={{ color: 'var(--ink-muted)' }}>
                      {item.group}
                    </span>
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Footer Shortcuts */}
        <div
          className="px-3.5 py-2 border-t flex items-center justify-between text-[10.5px] font-mono-num"
          style={{
            borderColor: 'var(--border-hairline)',
            backgroundColor: 'var(--bg-panel-subtle)',
            color: 'var(--ink-muted)',
          }}
        >
          <span>↑↓ Navigate · ↵ Select · ESC Close</span>
          <span>⌘K / Ctrl+K anytime</span>
        </div>
      </div>
    </div>
  );
};
