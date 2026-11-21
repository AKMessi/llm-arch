import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Blocks, Layers } from 'lucide-react';
import { BlueprintCanvas } from './components/BlueprintCanvas';
import { ComponentPalette } from './components/ComponentPalette';
import { DragOverlay } from './components/DragOverlay';
import { HeaderBar } from './components/HeaderBar';
import { TelemetryDock } from './components/TelemetryDock';
import { evaluateAllChallenges, getChallengeStarterConfig } from './data/challenges';
import { getComponentsByStage } from './data/components';
import rawGalleryModels from './data/galleryModels.json';
import { galleryModelToConfig, PRESET_BLUEPRINTS } from './data/presets';
import { decodeConfigFromHash, encodeConfigToHash } from './engine/codegen';
import { trackEvent } from './engine/analytics';
import { runArchitectureDiagnostics } from './engine/diagnostics';
import { computePhysics } from './engine/physics';
import { matchGalleryModels } from './engine/similarity';
import type {
  ArchitectureConfig,
  ArchitectureDimensions,
  ComponentDef,
  GalleryModel,
  PresetBlueprint,
  SavedBlueprint,
  StageId,
  WorkspaceTab,
} from './types/architecture';

const CodeExportView = lazy(() =>
  import('./components/CodeExportView').then((m) => ({ default: m.CodeExportView }))
);
const CommandPalette = lazy(() =>
  import('./components/CommandPalette').then((m) => ({ default: m.CommandPalette }))
);
const CompareView = lazy(() =>
  import('./components/CompareView').then((m) => ({ default: m.CompareView }))
);
const GalleryModal = lazy(() =>
  import('./components/GalleryModal').then((m) => ({ default: m.GalleryModal }))
);
const MissionsView = lazy(() =>
  import('./components/MissionsView').then((m) => ({ default: m.MissionsView }))
);
const SimulationLab = lazy(() =>
  import('./components/SimulationLab').then((m) => ({ default: m.SimulationLab }))
);
const SpecCardModal = lazy(() =>
  import('./components/SpecCardModal').then((m) => ({ default: m.SpecCardModal }))
);

const GALLERY_MODELS = rawGalleryModels as GalleryModel[];
const STORAGE_SOLVED = 'architecton_solved_missions_v1';
const STORAGE_VAULT = 'architecton_saved_vault_v1';
const STORAGE_QUICKSTART = 'architecton_hide_quickstart_v1';

interface DragState {
  comp: ComponentDef;
  startX: number;
  startY: number;
  x: number;
  y: number;
  isDragging: boolean;
}

function getInitialConfig(): ArchitectureConfig {
  const hashCfg = decodeConfigFromHash(window.location.hash);
  if (hashCfg) return hashCfg;
  return {
    ...PRESET_BLUEPRINTS[0].config,
    dims: { ...PRESET_BLUEPRINTS[0].config.dims },
  };
}

function loadJsonStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const [config, setConfig] = useState<ArchitectureConfig>(getInitialConfig);
  const [history, setHistory] = useState<ArchitectureConfig[]>([]);
  const [future, setFuture] = useState<ArchitectureConfig[]>([]);
  const [compareConfigB, setCompareConfigB] = useState<ArchitectureConfig>(() => ({
    ...PRESET_BLUEPRINTS[1].config,
    dims: { ...PRESET_BLUEPRINTS[1].config.dims },
  }));

  const [activeTab, setActiveTab] = useState<WorkspaceTab>('canvas');
  const [mobilePane, setMobilePane] = useState<'palette' | 'center' | 'telemetry'>('center');
  const [theme, setTheme] = useState<'paper' | 'obsidian'>('paper');
  const [stageFilter, setStageFilter] = useState<StageId | 'all'>('all');
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [cmdOpen, setCmdOpen] = useState(false);
  const [specCardOpen, setSpecCardOpen] = useState(false);
  const [flashedStage, setFlashedStage] = useState<StageId | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // Missions & Local Vault persistence
  const [solvedMissions, setSolvedMissions] = useState<Set<string>>(
    () => new Set(loadJsonStorage<string[]>(STORAGE_SOLVED, []))
  );
  const [pinnedMissionId, setPinnedMissionId] = useState<string | null>(null);
  const [savedBlueprints, setSavedBlueprints] = useState<SavedBlueprint[]>(() =>
    loadJsonStorage<SavedBlueprint[]>(STORAGE_VAULT, [])
  );
  const [showQuickStart, setShowQuickStart] = useState<boolean>(
    () => !loadJsonStorage<boolean>(STORAGE_QUICKSTART, false)
  );

  // Silky custom Pointer Drag-and-Drop state
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [hoveredDropStage, setHoveredDropStage] = useState<StageId | null>(null);
  const dropZonesRef = useRef<Partial<Record<StageId, HTMLDivElement | null>>>({});

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast((cur) => (cur === msg ? null : cur)), 2500);
  }, []);

  const commitConfig = useCallback((next: ArchitectureConfig) => {
    setConfig((prev) => {
      setHistory((h) => [...h.slice(-29), prev]);
      setFuture([]);
      return next;
    });
  }, []);

  const handleUndo = useCallback(() => {
    setHistory((h) => {
      if (h.length === 0) return h;
      const previous = h[h.length - 1];
      setConfig((cur) => {
        setFuture((f) => [cur, ...f.slice(0, 29)]);
        return previous;
      });
      return h.slice(0, -1);
    });
  }, []);

  const handleRedo = useCallback(() => {
    setFuture((f) => {
      if (f.length === 0) return f;
      const next = f[0];
      setConfig((cur) => {
        setHistory((h) => [...h.slice(-29), cur]);
        return next;
      });
      return f.slice(1);
    });
  }, []);

  // Global keyboard shortcuts: Ctrl+K / ⌘K, Ctrl+Z / Ctrl+Y (respecting native text inputs)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setCmdOpen((o) => !o);
        return;
      }
      const tag = (document.activeElement?.tagName ?? '').toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      } else if (
        ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'y') ||
        ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'z')
      ) {
        e.preventDefault();
        handleRedo();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleUndo, handleRedo]);

  // Equip a dragged or snapped component with smart dimension harmonization
  const equipComponent = useCallback((comp: ComponentDef) => {
    setConfig((prev) => {
      const nextDims: ArchitectureDimensions = { ...prev.dims };

      const enteringMoE =
        comp.stage === 'ffn' &&
        (comp.id === 'ffn_moe_sparse' ||
          comp.id === 'ffn_moe_shared' ||
          comp.id === 'ffn_latent_moe');
      const leavingMoE = comp.stage === 'ffn' && !enteringMoE && prev.dims.nExperts > 1;

      if (enteringMoE && prev.dims.nExperts <= 1) {
        nextDims.nExperts = comp.id === 'ffn_moe_shared' ? 128 : 64;
        nextDims.activeExperts = comp.id === 'ffn_moe_shared' ? 8 : 6;
        nextDims.nSharedExperts = comp.id === 'ffn_moe_shared' ? 1 : 0;
        nextDims.densePrefixLayers = comp.id === 'ffn_moe_shared' ? 2 : 0;
        nextDims.dFf = Math.min(2048, Math.max(1024, Math.round(prev.dims.dModel / 2)));
      } else if (leavingMoE) {
        nextDims.nExperts = 1;
        nextDims.activeExperts = 1;
        nextDims.nSharedExperts = 0;
        nextDims.densePrefixLayers = 0;
        nextDims.dFf = Math.round((prev.dims.dModel * 3.5) / 256) * 256;
      }

      if (comp.id === 'ffn_looped_depth' && nextDims.loopPasses < 2) {
        nextDims.loopPasses = 2;
      } else if (comp.stage === 'ffn' && comp.id !== 'ffn_looped_depth') {
        nextDims.loopPasses = 1;
      }

      if (comp.id === 'hyb_mamba2') {
        nextDims.hybridRatio = 5;
      } else if (comp.id === 'hyb_deltanet' || comp.id === 'hyb_kda') {
        nextDims.hybridRatio = 3;
      }

      setHistory((h) => [...h.slice(-29), prev]);
      setFuture([]);
      return {
        ...prev,
        [comp.stage]: comp.id,
        dims: nextDims,
      };
    });

    setFlashedStage(comp.stage);
    setTimeout(() => setFlashedStage((cur) => (cur === comp.stage ? null : cur)), 560);
    trackEvent('block_equipped', { stage: comp.stage, block_id: comp.id, block_name: comp.shortName });
  }, []);

  const handleStartDrag = useCallback((comp: ComponentDef, e: React.PointerEvent) => {
    if (e.button !== 0) return;
    setDragState({
      comp,
      startX: e.clientX,
      startY: e.clientY,
      x: e.clientX,
      y: e.clientY,
      isDragging: false,
    });
  }, []);

  useEffect(() => {
    if (!dragState) return;

    const findStageUnderPointer = (x: number, y: number): StageId | null => {
      for (const [stage, el] of Object.entries(dropZonesRef.current) as [
        StageId,
        HTMLDivElement | null,
      ][]) {
        if (!el) continue;
        const rect = el.getBoundingClientRect();
        if (
          x >= rect.left - 12 &&
          x <= rect.right + 12 &&
          y >= rect.top - 12 &&
          y <= rect.bottom + 12
        ) {
          return stage;
        }
      }
      return null;
    };

    const onPointerMove = (e: PointerEvent) => {
      const dist = Math.hypot(e.clientX - dragState.startX, e.clientY - dragState.startY);
      const isNowDragging = dragState.isDragging || dist > 6;
      setDragState((prev) =>
        prev
          ? {
              ...prev,
              x: e.clientX,
              y: e.clientY,
              isDragging: isNowDragging,
            }
          : null
      );
      if (isNowDragging) {
        if (activeTab !== 'canvas') setActiveTab('canvas');
        const hit = findStageUnderPointer(e.clientX, e.clientY);
        setHoveredDropStage(hit === dragState.comp.stage ? hit : null);
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!dragState.isDragging) {
        equipComponent(dragState.comp);
        showToast(`Snapped ${dragState.comp.shortName} into Blueprint`);
        setMobilePane('center');
      } else {
        const hit = findStageUnderPointer(e.clientX, e.clientY);
        if (hit === dragState.comp.stage) {
          equipComponent(dragState.comp);
          showToast(`Dropped ${dragState.comp.shortName} into ${dragState.comp.stage.toUpperCase()}`);
        } else {
          const targetEl = dropZonesRef.current[dragState.comp.stage];
          if (targetEl) {
            const r = targetEl.getBoundingClientRect();
            const withinCanvasX = e.clientX >= r.left - 40 && e.clientX <= r.right + 220;
            if (withinCanvasX) {
              equipComponent(dragState.comp);
              showToast(`Snapped ${dragState.comp.shortName} into slot`);
            }
          }
        }
      }
      setDragState(null);
      setHoveredDropStage(null);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, [dragState, equipComponent, showToast, activeTab]);

  const registerDropZone = useCallback((stage: StageId, el: HTMLDivElement | null) => {
    dropZonesRef.current[stage] = el;
  }, []);

  const handleUpdateDims = useCallback((patch: Partial<ArchitectureDimensions>) => {
    setConfig((prev) => ({
      ...prev,
      dims: { ...prev.dims, ...patch },
    }));
  }, []);

  const handleLoadPreset = useCallback(
    (preset: PresetBlueprint) => {
      commitConfig({
        ...preset.config,
        dims: { ...preset.config.dims },
      });
      setActiveTab('canvas');
      setMobilePane('center');
      showToast(`Loaded ${preset.name} Blueprint`);
      trackEvent('preset_loaded', { preset_id: preset.id, preset_name: preset.name });
    },
    [commitConfig, showToast]
  );

  const handleCloneGalleryModel = useCallback(
    (model: GalleryModel) => {
      const next = galleryModelToConfig(model);
      commitConfig(next);
      setActiveTab('canvas');
      setMobilePane('center');
      showToast(`Cloned ${model.name} onto Canvas`);
      trackEvent('gallery_model_cloned', { model_id: model.id, model_name: model.name });
    },
    [commitConfig, showToast]
  );

  const handleCompareModel = useCallback((model: GalleryModel) => {
    setCompareConfigB(galleryModelToConfig(model));
    setActiveTab('compare');
    setMobilePane('center');
  }, []);

  const handleStartMission = useCallback(
    (missionId: string, starterPresetId: string) => {
      const starter = getChallengeStarterConfig(starterPresetId);
      commitConfig(starter);
      setPinnedMissionId(missionId);
      setActiveTab('canvas');
      setMobilePane('center');
      showToast('Mission pinned! Drag blocks to satisfy all constraints.');
    },
    [commitConfig, showToast]
  );

  const handleSaveBlueprint = useCallback(() => {
    const entry: SavedBlueprint = {
      id: `bp-${Date.now()}`,
      savedAt: new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
      config: { ...config, dims: { ...config.dims } },
    };
    setSavedBlueprints((prev) => {
      const next = [entry, ...prev].slice(0, 15);
      localStorage.setItem(STORAGE_VAULT, JSON.stringify(next));
      return next;
    });
    showToast(`Saved "${config.name}" to Local Vault`);
  }, [config, showToast]);

  const handleDeleteSavedBlueprint = useCallback((id: string) => {
    setSavedBlueprints((prev) => {
      const next = prev.filter((b) => b.id !== id);
      localStorage.setItem(STORAGE_VAULT, JSON.stringify(next));
      return next;
    });
  }, []);

  const handleRandomize = useCallback(() => {
    const pick = <T,>(arr: T[]) => arr[Math.floor(Math.random() * arr.length)];
    const tok = pick(getComponentsByStage('tokenizer'));
    const pos = pick(getComponentsByStage('position'));
    const norm = pick(getComponentsByStage('norm'));
    const attn = pick(getComponentsByStage('attention'));
    const hyb = pick(getComponentsByStage('hybrid'));
    const ffn = pick(getComponentsByStage('ffn'));
    const out = pick(getComponentsByStage('output'));
    const isMoE =
      ffn.id === 'ffn_moe_sparse' || ffn.id === 'ffn_moe_shared' || ffn.id === 'ffn_latent_moe';
    const dModel = pick([2048, 3072, 4096, 5120, 6144]);
    const nLayers = pick([24, 32, 40, 48, 60]);
    const nHeads = Math.max(16, Math.round(dModel / 128));

    commitConfig({
      name: `Experimental ${attn.shortName}-${ffn.shortName}`,
      tokenizer: tok.id,
      position: pos.id,
      norm: norm.id,
      attention: attn.id,
      hybrid: hyb.id,
      ffn: ffn.id,
      output: out.id,
      dims: {
        ...config.dims,
        dModel,
        nLayers,
        nHeads,
        nKvHeads: Math.min(8, nHeads / 4),
        dFf: isMoE ? 1536 : Math.round((dModel * 3.5) / 256) * 256,
        nExperts: isMoE ? pick([64, 128, 256]) : 1,
        activeExperts: isMoE ? pick([4, 6, 8]) : 1,
        nSharedExperts: ffn.id === 'ffn_moe_shared' ? 1 : 0,
        densePrefixLayers: ffn.id === 'ffn_moe_shared' ? 2 : 0,
        hybridRatio: hyb.id === 'hyb_mamba2' ? 5 : pick([3, 5]),
        loopPasses: ffn.id === 'ffn_looped_depth' ? 2 : 1,
        contextLength: pick([65536, 131072, 262144, 1048576]),
      },
    });
    showToast(`Synthesized ${attn.shortName} + ${hyb.shortName} stack!`);
  }, [commitConfig, config.dims, showToast]);

  const handleCopyShareUrl = useCallback(() => {
    const encoded = encodeConfigToHash(config);
    const url = `${window.location.origin}${window.location.pathname}#bp=${encoded}`;
    window.history.replaceState(null, '', `#bp=${encoded}`);
    navigator.clipboard.writeText(url).catch(() => {});
    showToast('Shareable blueprint URL copied!');
  }, [config, showToast]);

  // Core Physics, Diagnostics, Gallery Radar & Live Mission Evaluator
  const physics = useMemo(() => computePhysics(config), [config]);
  const diagnostics = useMemo(
    () => runArchitectureDiagnostics(config, physics),
    [config, physics]
  );
  const allMatches = useMemo(
    () => matchGalleryModels(config, physics, GALLERY_MODELS, GALLERY_MODELS.length),
    [config, physics]
  );
  const similarityMap = useMemo(() => {
    const map = new Map<string, (typeof allMatches)[number]>();
    allMatches.forEach((m) => map.set(m.model.id, m));
    return map;
  }, [allMatches]);

  const evaluatedMissions = useMemo(
    () => evaluateAllChallenges(config, physics, diagnostics),
    [config, physics, diagnostics]
  );

  // Persist newly cleared missions automatically with celebration toast!
  useEffect(() => {
    const newlyCleared = evaluatedMissions.filter(
      (ev) => ev.completed && !solvedMissions.has(ev.challenge.id)
    );
    if (newlyCleared.length > 0) {
      setSolvedMissions((prev) => {
        const next = new Set(prev);
        newlyCleared.forEach((ev) => next.add(ev.challenge.id));
        localStorage.setItem(STORAGE_SOLVED, JSON.stringify(Array.from(next)));
        return next;
      });
      showToast(`🏆 Mission Cleared: ${newlyCleared[0].challenge.title}!`);
      newlyCleared.forEach((ev) =>
        trackEvent('mission_cleared', {
          mission_id: ev.challenge.id,
          mission_title: ev.challenge.title,
        })
      );
    }
  }, [evaluatedMissions, solvedMissions, showToast]);

  const pinnedMission = useMemo(
    () => evaluatedMissions.find((e) => e.challenge.id === pinnedMissionId) ?? null,
    [evaluatedMissions, pinnedMissionId]
  );

  const showPaletteOnDesktop = activeTab === 'canvas' || activeTab === 'lab';

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      <HeaderBar
        config={config}
        onRename={(name) => setConfig((prev) => ({ ...prev, name }))}
        activeTab={activeTab}
        onTabChange={(t) => {
          setActiveTab(t);
          setMobilePane('center');
        }}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'paper' ? 'obsidian' : 'paper'))}
        onLoadPreset={handleLoadPreset}
        onOpenGallery={() => setGalleryOpen(true)}
        onOpenCommandPalette={() => setCmdOpen(true)}
        onOpenSpecCard={() => setSpecCardOpen(true)}
        onRandomize={handleRandomize}
        onUndo={handleUndo}
        onRedo={handleRedo}
        canUndo={history.length > 0}
        canRedo={future.length > 0}
        galleryCount={GALLERY_MODELS.length}
        solvedMissionsCount={solvedMissions.size}
        totalMissionsCount={evaluatedMissions.length}
        savedBlueprints={savedBlueprints}
        onSaveBlueprint={handleSaveBlueprint}
        onLoadSavedBlueprint={(bp) => {
          commitConfig({ ...bp.config, dims: { ...bp.config.dims } });
          setActiveTab('canvas');
          showToast(`Loaded "${bp.config.name}" from Vault`);
        }}
        onDeleteSavedBlueprint={handleDeleteSavedBlueprint}
      />

      {/* Responsive Mobile / Tablet Pane Bar (< lg) */}
      <div
        className="lg:hidden shrink-0 border-b px-3 py-1.5 flex items-center justify-center gap-1.5"
        style={{ backgroundColor: 'var(--bg-panel-subtle)', borderColor: 'var(--border-hairline)' }}
      >
        {[
          { id: 'palette' as const, label: 'Blocks (49)', icon: <Layers className="w-3.5 h-3.5" /> },
          { id: 'center' as const, label: 'Workspace', icon: <Blocks className="w-3.5 h-3.5" /> },
          { id: 'telemetry' as const, label: 'Telemetry', icon: <Activity className="w-3.5 h-3.5" /> },
        ].map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setMobilePane(p.id)}
            className="flex-1 flex items-center justify-center gap-1.5 py-1 rounded text-[11.5px] font-semibold cursor-pointer"
            style={
              mobilePane === p.id
                ? { backgroundColor: 'var(--bg-panel)', color: 'var(--ink-primary)', boxShadow: 'var(--shadow-card)' }
                : { color: 'var(--ink-secondary)' }
            }
          >
            {p.icon}
            {p.label}
          </button>
        ))}
      </div>

      {/* Main 3-Pane Studio */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Component Library Rail */}
        {showPaletteOnDesktop && (
          <div className={`${mobilePane === 'palette' ? 'flex w-full lg:w-auto' : 'hidden lg:flex'} h-full shrink-0`}>
            <ComponentPalette
              config={config}
              selectedStageFilter={stageFilter}
              onStageFilterChange={setStageFilter}
              onStartDrag={handleStartDrag}
              onQuickEquip={(comp) => {
                equipComponent(comp);
                setMobilePane('center');
                showToast(`Snapped ${comp.shortName} into Blueprint`);
              }}
              draggingCompId={dragState?.isDragging ? dragState.comp.id : null}
            />
          </div>
        )}

        {/* Center Active Workspace */}
        <div className={`${mobilePane === 'center' ? 'flex' : 'hidden lg:flex'} flex-1 min-w-0 h-full`}>
          <Suspense
            fallback={
              <div className="flex-1 flex items-center justify-center blueprint-grid text-[12px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                Loading studio module…
              </div>
            }
          >
            {activeTab === 'canvas' && (
              <BlueprintCanvas
                config={config}
                physics={physics}
                theme={theme}
                draggingComp={dragState?.isDragging ? dragState.comp : null}
                hoveredDropStage={hoveredDropStage}
                flashedStage={flashedStage}
                pinnedMission={pinnedMission}
                onUnpinMission={() => setPinnedMissionId(null)}
                showQuickStart={showQuickStart}
                onDismissQuickStart={() => {
                  setShowQuickStart(false);
                  localStorage.setItem(STORAGE_QUICKSTART, 'true');
                }}
                onOpenMissions={() => setActiveTab('missions')}
                onOpenParetoLab={() => setActiveTab('lab')}
                onSelectStageFilter={(st) => setStageFilter(st)}
                onEquipComponent={(comp) => {
                  equipComponent(comp);
                  showToast(`Switched to ${comp.shortName}`);
                }}
                onUpdateDims={handleUpdateDims}
                registerDropZone={registerDropZone}
              />
            )}

            {activeTab === 'missions' && (
              <MissionsView
                evaluated={evaluatedMissions}
                solvedIds={solvedMissions}
                pinnedMissionId={pinnedMissionId}
                onPinMission={setPinnedMissionId}
                onStartMission={handleStartMission}
              />
            )}

            {activeTab === 'lab' && (
              <SimulationLab
                config={config}
                physics={physics}
                galleryModels={GALLERY_MODELS}
                onEquipComponent={(comp) => {
                  equipComponent(comp);
                  showToast(`Equipped ${comp.shortName}`);
                }}
                onCloneModel={handleCloneGalleryModel}
              />
            )}

            {activeTab === 'compare' && (
              <CompareView
                configA={config}
                physicsA={physics}
                configB={compareConfigB}
                onChangeConfigB={setCompareConfigB}
                onSnapshotAToB={() => {
                  setCompareConfigB({
                    ...config,
                    name: `${config.name} (Snapshot)`,
                    dims: { ...config.dims },
                  });
                  showToast('Snapshotted Blueprint A into Comparison Slot B');
                }}
                onCherryPickFromB={(comp) => {
                  equipComponent(comp);
                  showToast(`Cherry-picked ${comp.shortName} into Blueprint A!`);
                }}
                onReplaceAWithB={() => {
                  commitConfig({ ...compareConfigB, dims: { ...compareConfigB.dims } });
                  setActiveTab('canvas');
                  showToast(`Loaded ${compareConfigB.name} onto Canvas`);
                }}
                galleryModels={GALLERY_MODELS}
              />
            )}

            {activeTab === 'code' && (
              <CodeExportView
                config={config}
                physics={physics}
                onCopyShareUrl={handleCopyShareUrl}
                onOpenSpecCard={() => setSpecCardOpen(true)}
              />
            )}
          </Suspense>
        </div>

        {/* Right Live Telemetry & 120-Model Gallery Match Dock */}
        <div className={`${mobilePane === 'telemetry' ? 'flex w-full lg:w-auto' : 'hidden lg:flex'} h-full shrink-0`}>
          <TelemetryDock
            config={config}
            physics={physics}
            diagnostics={diagnostics}
            matches={allMatches}
            onApplySuggestion={(comp) => {
              equipComponent(comp);
              showToast(`Applied Advisor Upgrade: ${comp.shortName}`);
            }}
            onCloneGalleryModel={handleCloneGalleryModel}
            onCompareWithModel={handleCompareModel}
            onOpenGallery={() => setGalleryOpen(true)}
          />
        </div>
      </div>

      {/* Floating Custom Pointer Drag Ghost */}
      <DragOverlay
        comp={dragState?.isDragging ? dragState.comp : null}
        x={dragState?.x ?? 0}
        y={dragState?.y ?? 0}
        isOverValidTarget={hoveredDropStage !== null}
      />

      {/* Lazy Modals */}
      <Suspense fallback={null}>
        {galleryOpen && (
          <GalleryModal
            open={galleryOpen}
            onClose={() => setGalleryOpen(false)}
            models={GALLERY_MODELS}
            similarityMap={similarityMap}
            onLoadPreset={handleLoadPreset}
            onCloneModel={handleCloneGalleryModel}
            onCompareModel={handleCompareModel}
          />
        )}

        {cmdOpen && (
          <CommandPalette
            open={cmdOpen}
            onClose={() => setCmdOpen(false)}
            galleryModels={GALLERY_MODELS}
            onEquipComponent={(comp) => {
              equipComponent(comp);
              setActiveTab('canvas');
              showToast(`Snapped ${comp.shortName} via ⌘K`);
            }}
            onCloneModel={handleCloneGalleryModel}
            onStartMission={handleStartMission}
            onExportSpecCard={() => setSpecCardOpen(true)}
            onSaveBlueprint={handleSaveBlueprint}
            onRandomize={handleRandomize}
            onToggleTheme={() => setTheme((t) => (t === 'paper' ? 'obsidian' : 'paper'))}
            onOpenGallery={() => setGalleryOpen(true)}
          />
        )}

        {specCardOpen && (
          <SpecCardModal
            open={specCardOpen}
            onClose={() => setSpecCardOpen(false)}
            config={config}
            physics={physics}
            topMatch={allMatches[0]}
            defaultTheme={theme}
            onCopyShareUrl={handleCopyShareUrl}
          />
        )}
      </Suspense>

      {/* Crisp Floating Status Toast */}
      {toast && (
        <div
          className="fixed bottom-4 left-1/2 -translate-x-1/2 z-[95] px-3.5 py-2 rounded-lg border text-[12px] font-medium font-mono-num flex items-center gap-2 pointer-events-none"
          style={{
            backgroundColor: 'var(--ink-primary)',
            color: 'var(--ink-inverse)',
            borderColor: 'var(--border-strong)',
            boxShadow: 'var(--shadow-float)',
          }}
        >
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--stage-ffn)' }} />
          {toast}
        </div>
      )}
    </div>
  );
}
