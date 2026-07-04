/**
 * URL (de)serialization for the SAE selection on the /sae page.
 *
 * Three accepted URL formats, newest first:
 *   1. Multi:      ?model=<modelId>&saes=<saeId>,<saeId>,…
 *   2. Legacy:     ?modelId=<modelId>&saeId=<saeId>   (scatter-plot cross-links)
 *   3. Dimensions: ?model=&layer=&hookType=&width=    (old cascading selectors)
 */

import type { SaeModelInfo } from '@/lib/types/types';
import { parseSaeId } from '@/lib/utils/saeCollections';

export interface SaeSelection {
  modelId: string;
  saeIds: string[];
}

export function serializeSaesParam(saeIds: string[]): string {
  return saeIds.join(',');
}

export function parseSaesParam(param: string | null): string[] {
  if (!param) return [];
  return param
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

export interface SelectionUrlParams {
  saes: string | null;
  model: string | null;
  modelId: string | null;
  saeId: string | null;
  layer: string | null;
  hookType: string | null;
  width: string | null;
}

/**
 * Resolve the initial selection from URL params, or null when no
 * selection-related params are present (caller applies the default:
 * first model, all of its SAEs).
 */
export function resolveSelectionFromParams(
  params: SelectionUrlParams,
  models: SaeModelInfo[],
): SaeSelection | null {
  // Legacy single-SAE format (used by cross-links from the scatter plot)
  if (params.modelId && params.saeId) {
    return { modelId: params.modelId, saeIds: [params.saeId] };
  }

  // Multi-SAE format: model + comma-separated saeIds
  if (params.model && params.saes !== null) {
    return { modelId: params.model, saeIds: parseSaesParam(params.saes) };
  }

  // Old dimension format: filter the model's SAEs by layer/hook/width
  if (params.model || params.layer || params.hookType || params.width) {
    const modelId = params.model ?? models[0]?.modelId;
    if (!modelId) return null;
    const saeIds = models
      .filter((m) => m.modelId === modelId)
      .map((m) => m.saeId)
      .filter((saeId) => {
        const parsed = parseSaeId(saeId);
        if (params.layer !== null && String(parsed.layerIndex) !== params.layer) return false;
        if (params.hookType !== null && parsed.hookType !== params.hookType) return false;
        if (params.width !== null && parsed.width !== params.width) return false;
        return true;
      });
    return { modelId, saeIds };
  }

  return null;
}
