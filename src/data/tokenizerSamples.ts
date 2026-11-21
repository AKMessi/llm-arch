import type { ComponentDef } from '../types/architecture';

export interface TokenChunk {
  text: string;
  id: number;
  bytes: number;
}

export interface TokenizationResult {
  tokens: TokenChunk[];
  tokenCount: number;
  charCount: number;
  charsPerToken: number;
  effectiveCtxMultiplier: number;
}

export const SAMPLE_PROMPTS: { id: string; label: string; category: string; text: string }[] = [
  {
    id: 'code',
    label: 'PyTorch MLA Kernel',
    category: 'Code',
    text: `def forward(self, x: torch.Tensor, kv_cache=None):
    c_kv = self.w_dkv(x)  # low-rank latent [B, T, 512]
    k_rope = apply_rope(self.w_kr(x), freqs_cis)
    return flash_mla_attn(q, c_kv, k_rope, scale=1.0 / math.sqrt(128))`,
  },
  {
    id: 'prose',
    label: 'Architecture Essay',
    category: 'English',
    text: `Modern open-weight language models increasingly combine Multi-Head Latent Attention (MLA) or 3:1 Gated DeltaNet hybrids with fine-grained Mixture-of-Experts and Multi-Token Prediction to sustain million-token contexts at under 30 KiB per token.`,
  },
  {
    id: 'multilingual',
    label: 'Multilingual + Math',
    category: 'Polyglot',
    text: `Transformer scaling: KV_tok = 2·L·n_kv·d_h·bytes. 大语言模型架构设计 (LLM Architecture) — L'optimisation de la mémoire KV réduit la latence de 87.5% à 1,048,576 tokens!`,
  },
  {
    id: 'json',
    label: 'Agentic Tool Call',
    category: 'JSON',
    text: `{"tool":"architecture_eval","params":{"attention":"mla_dsa","n_layers":61,"kv_lora_rank":512,"active_experts":8,"total_experts":256,"context_length":1048576,"quantization":"fp8_e4m3"}}`,
  },
];

// Deterministic hash for stable synthetic token IDs inside the tokenizer's vocab range
function stableTokenId(str: string, vocabSize: number, salt: number): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (Math.abs(h) % Math.max(1000, vocabSize - 256)) + 256;
}

/**
 * High-fidelity tokenizer segmentation simulator reflecting real-world behavioral
 * differences across BPE 50k, tiktoken 128k, DeepSeek 129k (single-digit split),
 * Tekken 131k (tight code merges), Qwen 152k/248k (CJK & JSON merges), and Gemma 262k!
 */
export function simulateTokenization(text: string, tokenizer: ComponentDef): TokenizationResult {
  const family = tokenizer.tokenizerFamily ?? 'bpe_tiktoken';
  const vocabSize = tokenizer.vocabSize ?? 128256;
  const salt = family.split('').reduce((a, ch) => a + ch.charCodeAt(0), 0);

  if (!text) {
    return { tokens: [], tokenCount: 0, charCount: 0, charsPerToken: 0, effectiveCtxMultiplier: 1 };
  }

  const rawPieces: string[] = [];

  // Family-specific merge characteristics
  const splitsSingleDigits = family === 'bpe_deepseek' || family === 'sp_gemma';
  const mergesCodeIndent =
    family === 'bpe_tiktoken' ||
    family === 'bpe_tekken' ||
    family === 'bpe_qwen' ||
    family === 'bpe_qwen_wide';
  const mergesCjkPairs =
    family === 'bpe_qwen' ||
    family === 'bpe_qwen_wide' ||
    family === 'bpe_deepseek' ||
    family === 'sp_gemma';
  const wideVocab = family === 'bpe_qwen_wide' || family === 'sp_gemma';
  const smallVocab = family === 'bpe_gpt2' || family === 'byte_latent';

  let i = 0;
  while (i < text.length) {
    const ch = text[i];

    // Newlines
    if (ch === '\n') {
      rawPieces.push('\n');
      i++;
      continue;
    }

    // Spaces / indentation
    if (ch === ' ') {
      let j = i;
      while (j < text.length && text[j] === ' ') j++;
      const runLen = j - i;
      if (runLen >= 2) {
        if (mergesCodeIndent) {
          // Modern BPEs merge 4-space or 8-space indents into 1 token
          const step = runLen >= 4 ? 4 : runLen;
          rawPieces.push(text.slice(i, i + step));
          i += step;
        } else {
          // GPT-2 / 32k splits spaces much more finely
          rawPieces.push(' ');
          i++;
        }
        continue;
      }
      // Single leading space attaches to following English word in BPE/SentencePiece
      if (j < text.length && /[A-Za-zÀ-ÿ]/.test(text[j])) {
        let k = j;
        while (k < text.length && /[A-Za-zÀ-ÿ]/.test(text[k])) k++;
        const word = text.slice(i, k);
        pushWordPieces(word, smallVocab, wideVocab, rawPieces);
        i = k;
        continue;
      }
      rawPieces.push(' ');
      i++;
      continue;
    }

    // Digits
    if (/[0-9]/.test(ch)) {
      if (splitsSingleDigits) {
        // DeepSeek / Gemma split every digit individually for clean arithmetic
        rawPieces.push(ch);
        i++;
      } else if (family === 'bpe_tiktoken' || family === 'bpe_qwen' || family === 'bpe_qwen_wide' || family === 'bpe_tekken') {
        // Up to 3-digit grouping (cl100k / o200k / Qwen style)
        let j = i;
        while (j < text.length && /[0-9]/.test(text[j]) && j - i < 3) j++;
        rawPieces.push(text.slice(i, j));
        i = j;
      } else {
        // 1-2 digits for older BPE
        let j = i;
        while (j < text.length && /[0-9]/.test(text[j]) && j - i < 2) j++;
        rawPieces.push(text.slice(i, j));
        i = j;
      }
      continue;
    }

    // CJK characters
    if (/[\u4E00-\u9FFF\u3040-\u30FF]/.test(ch)) {
      if (mergesCjkPairs && i + 1 < text.length && /[\u4E00-\u9FFF\u3040-\u30FF]/.test(text[i + 1])) {
        rawPieces.push(text.slice(i, i + 2));
        i += 2;
      } else if (smallVocab) {
        // Byte-level 50k/32k splits UTF-8 CJK into 2-3 byte fragments!
        rawPieces.push(ch + '⟨b0⟩');
        rawPieces.push('⟨b1⟩');
        i++;
      } else {
        rawPieces.push(ch);
        i++;
      }
      continue;
    }

    // Alphabetic words & snake_case identifiers
    if (/[A-Za-zÀ-ÿ_]/.test(ch)) {
      let j = i;
      while (j < text.length && /[A-Za-zÀ-ÿ]/.test(text[j])) j++;
      if (j === i) {
        rawPieces.push(ch);
        i++;
      } else {
        const word = text.slice(i, j);
        pushWordPieces(word, smallVocab, wideVocab, rawPieces);
        i = j;
      }
      continue;
    }

    // JSON / code multi-char punctuation merges in modern wide tokenizers
    if (
      (wideVocab || family === 'bpe_tekken' || family === 'bpe_qwen') &&
      i + 1 < text.length &&
      ['":', ',"', '":', '()', '):', '->'].includes(text.slice(i, i + 2))
    ) {
      rawPieces.push(text.slice(i, i + 2));
      i += 2;
      continue;
    }

    rawPieces.push(ch);
    i++;
  }

  const tokens: TokenChunk[] = rawPieces.map((piece) => ({
    text: piece,
    id: stableTokenId(piece, vocabSize, salt),
    bytes: new TextEncoder().encode(piece.replace(/⟨b\d⟩/g, '')).length || 1,
  }));

  const charsPerToken = text.length / Math.max(1, tokens.length);
  // Relative to a GPT-2 50k baseline (~3.6 chars/token)
  const effectiveCtxMultiplier = Number((charsPerToken / 3.55).toFixed(2));

  return {
    tokens,
    tokenCount: tokens.length,
    charCount: text.length,
    charsPerToken: Number(charsPerToken.toFixed(2)),
    effectiveCtxMultiplier,
  };
}

function pushWordPieces(word: string, smallVocab: boolean, wideVocab: boolean, out: string[]) {
  const cleanLen = word.trim().length;
  const maxSingle = wideVocab ? 13 : smallVocab ? 7 : 10;
  if (cleanLen <= maxSingle) {
    out.push(word);
    return;
  }
  // Subword split preserving realistic stem + suffix boundaries
  const splitAt = wideVocab ? Math.ceil(word.length * 0.65) : Math.ceil(word.length * 0.55);
  out.push(word.slice(0, splitAt));
  const rest = word.slice(splitAt);
  if (smallVocab && rest.length > 5) {
    const mid = Math.ceil(rest.length / 2);
    out.push(rest.slice(0, mid));
    out.push(rest.slice(mid));
  } else {
    out.push(rest);
  }
}
