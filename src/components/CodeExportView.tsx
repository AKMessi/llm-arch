import React, { useMemo, useState } from 'react';
import { Check, Code2, Copy, Download, FileJson, Image as ImageIcon, Share2 } from 'lucide-react';
import {
  generateHuggingFaceConfig,
  generatePyTorchCode,
} from '../engine/codegen';
import { trackEvent } from '../engine/analytics';
import { formatKiB, formatParams } from '../engine/physics';
import type { ArchitectureConfig, PhysicsReport } from '../types/architecture';

interface CodeExportViewProps {
  config: ArchitectureConfig;
  physics: PhysicsReport;
  onCopyShareUrl: () => void;
  onOpenSpecCard: () => void;
}

type CodeTab = 'pytorch' | 'hf_config' | 'spec_json';

export const CodeExportView: React.FC<CodeExportViewProps> = ({
  config,
  physics,
  onCopyShareUrl,
  onOpenSpecCard,
}) => {
  const [tab, setTab] = useState<CodeTab>('pytorch');
  const [copied, setCopied] = useState(false);

  const pytorchCode = useMemo(() => generatePyTorchCode(config, physics), [config, physics]);
  const hfConfig = useMemo(() => generateHuggingFaceConfig(config, physics), [config, physics]);
  const specJson = useMemo(() => JSON.stringify(config, null, 2), [config]);

  const activeContent =
    tab === 'pytorch' ? pytorchCode : tab === 'hf_config' ? hfConfig : specJson;
  const activeFilename =
    tab === 'pytorch'
      ? `${config.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_model.py`
      : tab === 'hf_config'
        ? 'config.json'
        : 'architecton_blueprint.json';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(activeContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
    trackEvent('code_copied', { format: tab });
  };

  const handleDownload = () => {
    const blob = new Blob([activeContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = activeFilename;
    a.click();
    URL.revokeObjectURL(url);
    trackEvent('code_downloaded', { format: tab });
  };

  return (
    <main className="flex-1 h-full overflow-y-auto blueprint-grid p-4">
      <div className="max-w-[1040px] mx-auto space-y-3">
        {/* Top Export Bar */}
        <div
          className="rounded-lg border p-3.5 flex flex-wrap items-center justify-between gap-3"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div>
            <h2 className="font-display font-bold text-[14.5px]">
              Live PyTorch &amp; Hugging Face Blueprint Export
            </h2>
            <p className="text-[11.5px] font-mono-num mt-0.5" style={{ color: 'var(--ink-muted)' }}>
              {config.name} · {formatParams(physics.params.total)} ({formatParams(physics.params.active)} active) · {formatKiB(physics.kvKiBPerTokenBf16)}/tok BF16
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onOpenSpecCard}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11.5px] font-semibold cursor-pointer"
              style={{ backgroundColor: 'var(--stage-attention)', color: '#fff' }}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              Spec Card PNG
            </button>
            <button
              type="button"
              onClick={onCopyShareUrl}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-[11.5px] font-medium hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              <Share2 className="w-3.5 h-3.5" />
              Copy Blueprint URL
            </button>
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-[11.5px] font-medium hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              {copied ? <Check className="w-3.5 h-3.5" style={{ color: 'var(--stage-ffn)' }} /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied' : 'Copy Code'}
            </button>
            <button
              type="button"
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11.5px] font-semibold cursor-pointer"
              style={{ backgroundColor: 'var(--ink-primary)', color: 'var(--ink-inverse)' }}
            >
              <Download className="w-3.5 h-3.5" />
              Download {activeFilename}
            </button>
          </div>
        </div>

        {/* Code Viewer Card */}
        <div
          className="rounded-lg border overflow-hidden"
          style={{
            borderColor: 'var(--border-strong)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          {/* Tabs Header */}
          <div
            className="px-3 py-2 border-b flex items-center justify-between"
            style={{
              backgroundColor: 'var(--bg-panel)',
              borderColor: 'var(--border-hairline)',
            }}
          >
            <div className="flex items-center gap-1.5">
              {[
                { id: 'pytorch' as const, label: 'model.py (PyTorch nn.Module)', icon: <Code2 className="w-3.5 h-3.5" /> },
                { id: 'hf_config' as const, label: 'config.json (Hugging Face)', icon: <FileJson className="w-3.5 h-3.5" /> },
                { id: 'spec_json' as const, label: 'blueprint.json', icon: <FileJson className="w-3.5 h-3.5" /> },
              ].map((t) => {
                const active = tab === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setTab(t.id)}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11.5px] font-mono-num font-medium cursor-pointer transition-colors"
                    style={
                      active
                        ? {
                            backgroundColor: 'var(--ink-primary)',
                            color: 'var(--ink-inverse)',
                          }
                        : { color: 'var(--ink-secondary)' }
                    }
                  >
                    {t.icon}
                    {t.label}
                  </button>
                );
              })}
            </div>
            <span className="text-[10.5px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
              {activeContent.split('\n').length} lines
            </span>
          </div>

          {/* Code Body */}
          <pre
            className="p-4 m-0 overflow-x-auto text-[12px] leading-[1.6] font-mono-num"
            style={{
              backgroundColor: 'var(--bg-code)',
              color: '#e6edf7',
            }}
          >
            <code>{activeContent}</code>
          </pre>
        </div>
      </div>
    </main>
  );
};
