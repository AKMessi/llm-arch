import React, { useEffect, useState } from 'react';
import { Check, Copy, Download, Image as ImageIcon, Share2, X } from 'lucide-react';
import { classifyArchetype } from '../data/challenges';
import { trackEvent } from '../engine/analytics';
import { renderSpecCardCanvas } from '../engine/specCardRenderer';
import type {
  ArchitectureConfig,
  PhysicsReport,
  SimilarityMatch,
} from '../types/architecture';

interface SpecCardModalProps {
  open: boolean;
  onClose: () => void;
  config: ArchitectureConfig;
  physics: PhysicsReport;
  topMatch: SimilarityMatch | undefined;
  defaultTheme: 'paper' | 'obsidian';
  onCopyShareUrl: () => void;
}

export const SpecCardModal: React.FC<SpecCardModalProps> = ({
  open,
  onClose,
  config,
  physics,
  topMatch,
  defaultTheme,
  onCopyShareUrl,
}) => {
  const [cardTheme, setCardTheme] = useState<'paper' | 'obsidian'>(defaultTheme);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [copiedImg, setCopiedImg] = useState(false);

  useEffect(() => {
    setCardTheme(defaultTheme);
  }, [defaultTheme, open]);

  useEffect(() => {
    if (!open) return;
    const canvas = renderSpecCardCanvas(config, physics, topMatch, cardTheme);
    setPreviewUrl(canvas.toDataURL('image/png'));
  }, [open, config, physics, topMatch, cardTheme]);

  if (!open) return null;

  const archetype = classifyArchetype(config, physics);

  const handleDownloadPng = () => {
    const canvas = renderSpecCardCanvas(config, physics, topMatch, cardTheme);
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = `${config.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-spec-card.png`;
    a.click();
    trackEvent('spec_card_downloaded', { theme: cardTheme, blueprint: config.name });
  };

  const handleCopyImage = async () => {
    try {
      const canvas = renderSpecCardCanvas(config, physics, topMatch, cardTheme);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob((b) => resolve(b), 'image/png')
      );
      if (blob && typeof ClipboardItem !== 'undefined') {
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
        setCopiedImg(true);
        setTimeout(() => setCopiedImg(false), 1800);
      } else {
        handleDownloadPng();
      }
    } catch {
      handleDownloadPng();
    }
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className="w-full max-w-[920px] rounded-xl border overflow-hidden flex flex-col max-h-[92vh]"
        style={{
          backgroundColor: 'var(--bg-panel)',
          borderColor: 'var(--border-strong)',
          boxShadow: 'var(--shadow-float)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="px-5 py-3.5 border-b flex flex-wrap items-center justify-between gap-3"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <div>
            <div className="flex items-center gap-2">
              <ImageIcon className="w-4 h-4" style={{ color: 'var(--stage-attention)' }} />
              <h2 className="font-display font-bold text-[15px]">
                Shareable Architecture Spec Card (2× Retina PNG)
              </h2>
            </div>
            <p className="text-[11.5px] mt-0.5" style={{ color: 'var(--ink-secondary)' }}>
              Archetype: <strong>{archetype.title}</strong> — sized 1200×675 (2400×1350px) for X, LinkedIn, Reddit &amp; Discord.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {/* Theme toggle specifically for the exported poster */}
            <div
              className="flex items-center p-0.5 rounded-md border text-[11px] font-medium"
              style={{
                backgroundColor: 'var(--bg-panel-subtle)',
                borderColor: 'var(--border-hairline)',
              }}
            >
              <button
                type="button"
                onClick={() => setCardTheme('paper')}
                className="px-2.5 py-1 rounded cursor-pointer"
                style={
                  cardTheme === 'paper'
                    ? { backgroundColor: 'var(--bg-panel)', color: 'var(--ink-primary)', boxShadow: 'var(--shadow-card)' }
                    : { color: 'var(--ink-secondary)' }
                }
              >
                Paper Poster
              </button>
              <button
                type="button"
                onClick={() => setCardTheme('obsidian')}
                className="px-2.5 py-1 rounded cursor-pointer"
                style={
                  cardTheme === 'obsidian'
                    ? { backgroundColor: 'var(--bg-panel)', color: 'var(--ink-primary)', boxShadow: 'var(--shadow-card)' }
                    : { color: 'var(--ink-secondary)' }
                }
              >
                Obsidian Poster
              </button>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-md border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Rendered Preview */}
        <div className="p-5 overflow-y-auto blueprint-grid flex items-center justify-center">
          {previewUrl && (
            <img
              src={previewUrl}
              alt={`${config.name} Architecture Spec Card`}
              className="w-full max-w-[820px] rounded-lg border"
              style={{ borderColor: 'var(--border-strong)', boxShadow: 'var(--shadow-float)' }}
            />
          )}
        </div>

        {/* Footer Actions */}
        <div
          className="px-5 py-3 border-t flex flex-wrap items-center justify-between gap-2"
          style={{ borderColor: 'var(--border-hairline)', backgroundColor: 'var(--bg-panel)' }}
        >
          <button
            type="button"
            onClick={onCopyShareUrl}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-[12px] font-medium hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
            style={{ borderColor: 'var(--border-hairline)' }}
          >
            <Share2 className="w-3.5 h-3.5" />
            Copy Interactive URL
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopyImage}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-[12px] font-medium hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-strong)' }}
            >
              {copiedImg ? (
                <>
                  <Check className="w-3.5 h-3.5" style={{ color: 'var(--stage-ffn)' }} />
                  Copied PNG!
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy PNG Image
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleDownloadPng}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-[12px] font-semibold cursor-pointer"
              style={{
                backgroundColor: 'var(--stage-attention)',
                color: '#fff',
              }}
            >
              <Download className="w-3.5 h-3.5" />
              Download Retina PNG
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
