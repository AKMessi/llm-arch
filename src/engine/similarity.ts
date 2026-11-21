import type {
  ArchitectureConfig,
  GalleryModel,
  PhysicsReport,
  SimilarityMatch,
} from '../types/architecture';

/**
 * Computes structural similarity (0..100%) between the user's live drag-and-drop
 * architecture and ALL 120 real open-weight models in Sebastian Raschka's
 * LLM Architecture Gallery!
 */
export function matchGalleryModels(
  config: ArchitectureConfig,
  physics: PhysicsReport,
  gallery: GalleryModel[],
  limit = 6
): SimilarityMatch[] {
  const userConcepts = new Set(physics.conceptTags);
  const isUserMoE =
    config.ffn === 'ffn_moe_sparse' ||
    config.ffn === 'ffn_moe_shared' ||
    config.ffn === 'ffn_latent_moe';
  const isUserRecurrent = config.hybrid === 'hyb_xlstm';
  const isUserMamba = config.hybrid === 'hyb_mamba2';
  const isUserHybrid = config.hybrid !== 'hyb_none' && config.hybrid !== 'hyb_swa';

  const scored: SimilarityMatch[] = gallery.map((model) => {
    const modelConcepts = new Set(model.concepts);

    // 1. Concept Jaccard similarity (weighted core architectural signature)
    const shared: string[] = [];
    const missingInUser: string[] = [];
    const extraInUser: string[] = [];

    for (const c of modelConcepts) {
      if (userConcepts.has(c)) shared.push(c);
      else missingInUser.push(c);
    }
    for (const c of userConcepts) {
      if (!modelConcepts.has(c)) extraInUser.push(c);
    }

    const unionSize = new Set([...userConcepts, ...modelConcepts]).size;
    let conceptScore = unionSize > 0 ? shared.length / unionSize : 0.3;

    // Special handling for models with empty concept arrays in models.yml (GPT-2, Nemotron 3, xLSTM)
    if (modelConcepts.size === 0) {
      const text = (model.name + ' ' + model.attention + ' ' + model.decoderType).toLowerCase();
      if (/gpt-2/.test(text)) {
        conceptScore =
          (config.attention === 'attn_mha' ? 0.35 : 0) +
          (config.position === 'pos_abs_learned' ? 0.35 : 0) +
          (config.norm === 'norm_pre_ln' ? 0.15 : 0) +
          (config.ffn === 'ffn_gelu' ? 0.15 : 0);
      } else if (/mamba-2/.test(text)) {
        conceptScore =
          (isUserMamba ? 0.65 : 0) +
          (isUserMoE && /moe/i.test(model.decoderType) ? 0.2 : 0) +
          (config.attention === 'attn_gqa' ? 0.15 : 0);
      } else if (/xlstm|mlstm/.test(text)) {
        conceptScore = isUserRecurrent ? 0.95 : 0.05;
      }
    }

    // 2. Decoder family structural match
    const modelIsMoE = /moe/i.test(model.decoderType);
    const modelIsHybrid = /hybrid/i.test(model.decoderType);
    const modelIsRecurrent = /recurrent/i.test(model.decoderType);

    let familyBonus = 0;
    if (isUserRecurrent && modelIsRecurrent) familyBonus = 0.25;
    else if (isUserMoE === modelIsMoE && isUserHybrid === modelIsHybrid) familyBonus = 0.18;
    else if (isUserMoE === modelIsMoE) familyBonus = 0.08;

    // 3. Attention family bonus
    const attnStr = model.attention.toLowerCase();
    let attnBonus = 0;
    if (config.attention === 'attn_mla' && /\bmla\b/.test(attnStr)) attnBonus = 0.12;
    else if (config.attention === 'attn_mla_dsa' && /sparse attention/.test(attnStr)) attnBonus = 0.15;
    else if (config.attention === 'attn_csa_hca' && /csa|hca/.test(attnStr)) attnBonus = 0.16;
    else if (config.attention === 'attn_gqa' && /\bgqa\b/.test(attnStr)) attnBonus = 0.08;
    else if (config.attention === 'attn_gated_gqa' && /gated/.test(attnStr)) attnBonus = 0.12;
    else if (config.attention === 'attn_mha' && /\bmha\b/.test(attnStr)) attnBonus = 0.12;
    else if (config.attention === 'attn_mqa' && /multi-query/.test(attnStr)) attnBonus = 0.14;

    // 4. KV-cache order-of-magnitude proximity bonus
    let kvBonus = 0;
    if (model.kvCacheKiB !== null) {
      if (model.kvCacheKiB === 0 && physics.kvKiBPerTokenBf16 === 0) {
        kvBonus = 0.1;
      } else if (model.kvCacheKiB > 0 && physics.kvKiBPerTokenBf16 > 0) {
        const logDiff = Math.abs(Math.log2(model.kvCacheKiB / physics.kvKiBPerTokenBf16));
        kvBonus = Math.max(0, 0.08 * (1 - logDiff / 4));
      }
    }

    // 5. Parameter scale proximity bonus (keeps 8B builds ranking closest to 8B models, 32B to 32B, etc.)
    let scaleBonus = 0;
    if (model.totalParamsB && model.totalParamsB > 0 && physics.params.total > 0) {
      const userB = physics.params.total / 1e9;
      const logScaleDiff = Math.abs(Math.log2(model.totalParamsB / userB));
      scaleBonus = Math.max(0, 0.09 * (1 - logScaleDiff / 3.5));
    }

    const rawScore = Math.min(
      0.99,
      conceptScore * 0.58 + familyBonus + attnBonus + kvBonus + scaleBonus
    );
    const score = Math.max(5, Math.round(rawScore * 100));

    return {
      model,
      score,
      sharedConcepts: shared,
      missingInUser,
      extraInUser,
    };
  });

  return scored
    .sort((a, b) => b.score - a.score || (b.model.date > a.model.date ? 1 : -1))
    .slice(0, limit);
}
