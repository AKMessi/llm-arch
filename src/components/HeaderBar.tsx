import React, { useState } from 'react';
import {
  Blocks,
  Bookmark,
  ChevronDown,
  Code2,
  FlaskConical,
  GitCompare,
  Image as ImageIcon,
  Layers,
  Moon,
  Redo2,
  Search,
  Shuffle,
  Sparkles,
  Star,
  Sun,
  Trash2,
  Trophy,
  Undo2,
} from 'lucide-react';
import { trackEvent } from '../engine/analytics';
import { PRESET_BLUEPRINTS } from '../data/presets';
import type {
  ArchitectureConfig,
  PresetBlueprint,
  SavedBlueprint,
  WorkspaceTab,
} from '../types/architecture';

interface HeaderBarProps {
  config: ArchitectureConfig;
  onRename: (name: string) => void;
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
  theme: 'paper' | 'obsidian';
  onToggleTheme: () => void;
  onLoadPreset: (preset: PresetBlueprint) => void;
  onOpenGallery: () => void;
  onOpenCommandPalette: () => void;
  onOpenSpecCard: () => void;
  onRandomize: () => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  galleryCount: number;
  solvedMissionsCount: number;
  totalMissionsCount: number;
  savedBlueprints: SavedBlueprint[];
  onSaveBlueprint: () => void;
  onLoadSavedBlueprint: (bp: SavedBlueprint) => void;
  onDeleteSavedBlueprint: (id: string) => void;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  config,
  onRename,
  activeTab,
  onTabChange,
  theme,
  onToggleTheme,
  onLoadPreset,
  onOpenGallery,
  onOpenCommandPalette,
  onOpenSpecCard,
  onRandomize,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  galleryCount,
  solvedMissionsCount,
  totalMissionsCount,
  savedBlueprints,
  onSaveBlueprint,
  onLoadSavedBlueprint,
  onDeleteSavedBlueprint,
}) => {
  const [presetMenuOpen, setPresetMenuOpen] = useState(false);

  const tabs: {
    id: WorkspaceTab;
    label: string;
    icon: React.ReactNode;
    badge?: string;
  }[] = [
    { id: 'canvas', label: 'Canvas', icon: <Blocks className="w-3.5 h-3.5" /> },
    {
      id: 'missions',
      label: 'Missions',
      icon: <Trophy className="w-3.5 h-3.5" />,
      badge: `${solvedMissionsCount}/${totalMissionsCount}`,
    },
    { id: 'lab', label: 'Lab', icon: <FlaskConical className="w-3.5 h-3.5" /> },
    { id: 'compare', label: 'Compare', icon: <GitCompare className="w-3.5 h-3.5" /> },
    { id: 'code', label: 'Export', icon: <Code2 className="w-3.5 h-3.5" /> },
  ];

  return (
    <header
      className="h-14 shrink-0 border-b flex items-center justify-between px-3 gap-2 select-none relative z-30"
      style={{
        backgroundColor: 'var(--bg-panel)',
        borderColor: 'var(--border-hairline)',
      }}
    >
      {/* Left: Brand Mark + Editable Name + Presets/Vault + Gallery */}
      <div className="flex items-center gap-2 min-w-0">
        <div className="flex items-center gap-2 shrink-0">
          <div
            className="w-8 h-8 rounded-md flex items-center justify-center border"
            style={{
              backgroundColor: 'var(--ink-primary)',
              color: 'var(--ink-inverse)',
              borderColor: 'var(--border-strong)',
            }}
          >
            <svg viewBox="0 0 24 24" className="w-4.5 h-4.5" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 18L12 4l8 14" strokeLinecap="square" />
              <path d="M7.5 13h9" strokeLinecap="square" />
              <circle cx="12" cy="18" r="1.5" fill="currentColor" stroke="none" />
            </svg>
          </div>
          <div className="leading-none hidden sm:block">
            <div className="font-display font-bold text-[12.5px] tracking-[0.07em] uppercase">
              Architecton
            </div>
            <div className="text-[9.5px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
              LLM STUDIO v1
            </div>
          </div>
        </div>

        {/* Editable Blueprint Name */}
        <input
          type="text"
          value={config.name}
          onChange={(e) => onRename(e.target.value)}
          aria-label="Blueprint Name"
          className="hidden 2xl:block text-[12.5px] font-medium px-2 py-1 rounded border border-transparent hover:border-[var(--border-hairline)] focus:border-[var(--border-strong)] focus:outline-none bg-transparent w-44 truncate transition-colors"
          style={{ color: 'var(--ink-primary)' }}
        />

        {/* Presets & Saved Vault Dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setPresetMenuOpen((o) => !o)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-[12px] font-medium transition-colors cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
            style={{
              borderColor: 'var(--border-hairline)',
              color: 'var(--ink-primary)',
            }}
          >
            <Sparkles className="w-3.5 h-3.5" style={{ color: 'var(--stage-attention)' }} />
            <span className="hidden sm:inline">Presets</span>
            {savedBlueprints.length > 0 && (
              <span
                className="text-[9.5px] font-mono-num px-1 rounded font-bold"
                style={{ backgroundColor: 'var(--stage-ffn-soft)', color: 'var(--stage-ffn)' }}
              >
                +{savedBlueprints.length}
              </span>
            )}
            <ChevronDown className="w-3 h-3 opacity-60" />
          </button>

          {presetMenuOpen && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setPresetMenuOpen(false)}
              />
              <div
                className="absolute left-0 top-full mt-1.5 w-[460px] max-w-[92vw] rounded-lg border p-2.5 z-50 space-y-2.5 max-h-[78vh] overflow-y-auto"
                style={{
                  backgroundColor: 'var(--bg-panel-elevated)',
                  borderColor: 'var(--border-strong)',
                  boxShadow: 'var(--shadow-float)',
                }}
              >
                {/* Saved Vault Section */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between px-1">
                    <span className="text-[10.5px] font-bold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
                      My Saved Vault ({savedBlueprints.length})
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        onSaveBlueprint();
                      }}
                      className="flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded cursor-pointer"
                      style={{ backgroundColor: 'var(--stage-ffn-soft)', color: 'var(--stage-ffn)' }}
                    >
                      <Bookmark className="w-3 h-3" />
                      Save Current
                    </button>
                  </div>

                  {savedBlueprints.length > 0 && (
                    <div className="space-y-1">
                      {savedBlueprints.map((bp) => (
                        <div
                          key={bp.id}
                          className="flex items-center justify-between gap-2 px-2 py-1.5 rounded-md border hover:border-[var(--border-strong)]"
                          style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
                        >
                          <button
                            type="button"
                            onClick={() => {
                              onLoadSavedBlueprint(bp);
                              setPresetMenuOpen(false);
                            }}
                            className="text-left flex-1 min-w-0 cursor-pointer"
                          >
                            <div className="text-[12px] font-semibold truncate">{bp.config.name}</div>
                            <div className="text-[9.5px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                              Saved {bp.savedAt}
                            </div>
                          </button>
                          <button
                            type="button"
                            onClick={() => onDeleteSavedBlueprint(bp.id)}
                            className="p-1 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
                            style={{ color: 'var(--ink-muted)' }}
                            title="Delete saved blueprint"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* 12 Iconic Presets */}
                <div className="border-t pt-2" style={{ borderColor: 'var(--border-hairline)' }}>
                  <div className="px-1 pb-1.5 flex items-center justify-between">
                    <span className="text-[10.5px] font-bold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
                      {PRESET_BLUEPRINTS.length} Landmark Reference Blueprints
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        setPresetMenuOpen(false);
                        onOpenGallery();
                      }}
                      className="text-[11px] font-medium underline cursor-pointer"
                      style={{ color: 'var(--stage-tokenizer)' }}
                    >
                      All {galleryCount} →
                    </button>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                    {PRESET_BLUEPRINTS.map((preset) => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          onLoadPreset(preset);
                          setPresetMenuOpen(false);
                        }}
                        className="text-left p-2 rounded-md border border-transparent hover:border-[var(--border-strong)] hover:bg-[var(--bg-panel-subtle)] transition-all cursor-pointer"
                      >
                        <div className="flex items-center justify-between gap-1">
                          <span className="text-[12px] font-semibold truncate">{preset.name}</span>
                          <span
                            className="text-[9.5px] font-mono-num px-1.5 py-0.5 rounded shrink-0"
                            style={{
                              backgroundColor: 'var(--stage-attention-soft)',
                              color: 'var(--stage-attention)',
                            }}
                          >
                            {preset.kvBadge}
                          </span>
                        </div>
                        <div className="text-[10px] font-mono-num mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                          {preset.company} · {preset.scaleBadge}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* 120-Model Gallery Trigger */}
        <button
          type="button"
          onClick={onOpenGallery}
          className="flex items-center gap-1.5 px-2 py-1.5 rounded-md border text-[12px] font-medium transition-colors cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
          style={{
            borderColor: 'var(--border-hairline)',
            color: 'var(--ink-primary)',
          }}
          title={`Explore all ${galleryCount} open-weight models in the Architecture Gallery`}
        >
          <Layers className="w-3.5 h-3.5" style={{ color: 'var(--stage-tokenizer)' }} />
          <span className="hidden xl:inline">Gallery</span>
          <span
            className="text-[10px] font-mono-num px-1.5 rounded-full font-semibold"
            style={{
              backgroundColor: 'var(--stage-tokenizer-soft)',
              color: 'var(--stage-tokenizer)',
            }}
          >
            {galleryCount}
          </span>
        </button>
      </div>

      {/* Center: Primary Workspace Tabs */}
      <nav
        className="flex items-center p-0.5 rounded-lg border overflow-x-auto"
        style={{
          backgroundColor: 'var(--bg-panel-subtle)',
          borderColor: 'var(--border-hairline)',
        }}
        aria-label="Workspace View"
      >
        {tabs.map((tab) => {
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onTabChange(tab.id)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[12px] font-medium transition-all cursor-pointer whitespace-nowrap"
              style={
                active
                  ? {
                      backgroundColor: 'var(--bg-panel)',
                      color: 'var(--ink-primary)',
                      boxShadow: 'var(--shadow-card)',
                    }
                  : {
                      color: 'var(--ink-secondary)',
                    }
              }
            >
              {tab.icon}
              <span className="hidden md:inline">{tab.label}</span>
              {tab.badge && (
                <span
                  className="text-[9.5px] font-mono-num font-bold px-1 rounded hidden lg:inline"
                  style={{
                    backgroundColor:
                      solvedMissionsCount === totalMissionsCount
                        ? 'var(--stage-ffn-soft)'
                        : 'var(--stage-hybrid-soft)',
                    color:
                      solvedMissionsCount === totalMissionsCount
                        ? 'var(--stage-ffn)'
                        : 'var(--stage-hybrid)',
                  }}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Right: Studio Utilities */}
      <div className="flex items-center gap-1.5 shrink-0">
        {/* Command Palette ⌘K */}
        <button
          type="button"
          onClick={onOpenCommandPalette}
          title="Command Palette (Ctrl+K / ⌘K)"
          className="hidden sm:flex items-center gap-1.5 px-2 py-1.5 rounded-md border text-[11px] font-mono-num hover:bg-[var(--bg-panel-subtle)] cursor-pointer transition-colors"
          style={{
            borderColor: 'var(--border-hairline)',
            color: 'var(--ink-muted)',
          }}
        >
          <Search className="w-3.5 h-3.5" />
          <span className="hidden xl:inline">⌘K</span>
        </button>

        {/* Undo / Redo */}
        <div
          className="hidden md:flex items-center rounded-md border overflow-hidden"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <button
            type="button"
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            className="p-1.5 hover:bg-[var(--bg-panel-subtle)] disabled:opacity-30 cursor-pointer disabled:cursor-default transition-colors"
          >
            <Undo2 className="w-3.5 h-3.5" />
          </button>
          <div className="w-px h-4" style={{ backgroundColor: 'var(--border-hairline)' }} />
          <button
            type="button"
            onClick={onRedo}
            disabled={!canRedo}
            title="Redo (Ctrl+Y)"
            className="p-1.5 hover:bg-[var(--bg-panel-subtle)] disabled:opacity-30 cursor-pointer disabled:cursor-default transition-colors"
          >
            <Redo2 className="w-3.5 h-3.5" />
          </button>
        </div>

        <button
          type="button"
          onClick={onRandomize}
          title="Synthesize Random Experimental Architecture"
          className="p-1.5 rounded-md border hover:bg-[var(--bg-panel-subtle)] transition-colors cursor-pointer"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <Shuffle className="w-3.5 h-3.5" />
        </button>

        {/* Viral Spec Card PNG Trigger */}
        <button
          type="button"
          onClick={onOpenSpecCard}
          title="Export 1200×675 Shareable Architecture Spec Card PNG"
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11.5px] font-semibold cursor-pointer transition-opacity hover:opacity-90"
          style={{
            backgroundColor: 'var(--stage-attention)',
            color: '#fff',
          }}
        >
          <ImageIcon className="w-3.5 h-3.5" />
          <span className="hidden lg:inline">Spec Card</span>
        </button>

        {/* GitHub Open-Source Star Link */}
        <a
          href="https://github.com/AKMessi/llm-arch"
          target="_blank"
          rel="noreferrer"
          onClick={() => trackEvent('github_star_clicked', { location: 'header' })}
          title="Star ARCHITECTON on GitHub (MIT Open Source)"
          className="hidden sm:flex items-center gap-1 px-2 py-1.5 rounded-md border text-[11.5px] font-medium hover:bg-[var(--bg-panel-subtle)] transition-colors"
          style={{
            borderColor: 'var(--border-hairline)',
            color: 'var(--ink-primary)',
          }}
        >
          <Star className="w-3.5 h-3.5" style={{ color: 'var(--stage-hybrid)' }} />
          <span className="hidden 2xl:inline">GitHub</span>
        </a>

        <button
          type="button"
          onClick={onToggleTheme}
          title={theme === 'paper' ? 'Switch to Obsidian Blueprint Mode' : 'Switch to Drafting Paper Mode'}
          className="p-1.5 rounded-md border hover:bg-[var(--bg-panel-subtle)] transition-colors cursor-pointer"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          {theme === 'paper' ? <Moon className="w-3.5 h-3.5" /> : <Sun className="w-3.5 h-3.5" />}
        </button>
      </div>
    </header>
  );
};
