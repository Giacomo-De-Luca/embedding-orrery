/**
 * Probe hyperparameter model + GraphQL input builder.
 *
 * Pure, UI-agnostic: the ProbeSection form holds a single ProbeParams object,
 * the settings popover edits the fields relevant to the active kind, and
 * `buildTrainProbeInput` turns (collection, field, kind, params) into the
 * `TrainProbeInput` variables — omitting anything left at its default so the
 * backend applies its own default (one source of truth).
 */

export type ProbeKind =
  | 'ridge'
  | 'lasso'
  | 'massmean'
  | 'massmean_cov'
  | 'svr'
  | 'logreg'
  | 'mlp';

export interface ProbeParams {
  alpha: number; // ridge / lasso L2/L1
  c: number; // svr / logreg inverse-regularisation
  kernel: 'rbf' | 'linear'; // svr
  classWeight: 'none' | 'balanced'; // logreg
  hiddenSize: number; // mlp single hidden layer width
  epochs: number; // mlp
  patience: number; // mlp early-stopping patience (dev-set based)
  seed: number; // shared
  trainSplit: number; // shared (train fraction)
  maxTrainSamples: number; // shared (training-pool subsample cap)
}

export const DEFAULT_PROBE_PARAMS: ProbeParams = {
  alpha: 1.0,
  c: 1.0,
  kernel: 'rbf',
  classWeight: 'none',
  hiddenSize: 256,
  epochs: 100,
  patience: 10,
  seed: 42, // mirrors the backend ProbeConfig default — displayed = trained
  trainSplit: 0.8,
  maxTrainSamples: 50000,
};

export const PROBE_KIND_OPTIONS: { value: ProbeKind; label: string }[] = [
  { value: 'ridge', label: 'Ridge (linear)' },
  { value: 'lasso', label: 'Lasso (sparse linear)' },
  { value: 'massmean', label: 'Mass-mean' },
  { value: 'massmean_cov', label: 'Mass-mean (cov-corrected)' },
  { value: 'svr', label: 'SVR (RBF)' },
  { value: 'logreg', label: 'Logistic (binary)' },
  { value: 'mlp', label: 'MLP (nonlinear)' },
];

/** Kinds that require a binary (two-class) target. */
export function isBinaryKind(kind: ProbeKind): boolean {
  return kind === 'logreg';
}

/** Which parameter controls the settings popover should show for a kind. */
export function probeParamFields(
  kind: ProbeKind,
): Array<'alpha' | 'c' | 'kernel' | 'classWeight' | 'hiddenSize' | 'epochs' | 'patience'> {
  switch (kind) {
    case 'ridge':
    case 'lasso':
      return ['alpha'];
    case 'svr':
      return ['c', 'kernel'];
    case 'logreg':
      return ['c', 'classWeight'];
    case 'mlp':
      return ['hiddenSize', 'epochs', 'patience'];
    case 'massmean':
    case 'massmean_cov':
    default:
      return [];
  }
}

export interface TrainProbeInputVars {
  collectionName: string;
  targetField: string;
  kind: ProbeKind;
  alpha?: number;
  c?: number;
  kernel?: string;
  classWeight?: string;
  hiddenDims?: number[];
  epochs?: number;
  patience?: number;
  seed?: number;
  trainSplit?: number;
  maxTrainSamples?: number;
}

/**
 * Build the `TrainProbeInput` variables for a probe run.
 *
 * Only the parameters that (a) apply to the kind and (b) differ from the
 * default are included, so the backend's defaults stay authoritative. Shared
 * params (seed, trainSplit) are always sent when non-default.
 */
export function buildTrainProbeInput(
  collectionName: string,
  targetField: string,
  kind: ProbeKind,
  params: ProbeParams,
): TrainProbeInputVars {
  const d = DEFAULT_PROBE_PARAMS;
  const vars: TrainProbeInputVars = { collectionName, targetField, kind };
  const fields = new Set(probeParamFields(kind));

  if (fields.has('alpha') && params.alpha !== d.alpha) vars.alpha = params.alpha;
  if (fields.has('c') && params.c !== d.c) vars.c = params.c;
  if (fields.has('kernel') && params.kernel !== d.kernel) vars.kernel = params.kernel;
  if (fields.has('classWeight') && params.classWeight !== 'none') {
    vars.classWeight = params.classWeight;
  }
  if (fields.has('hiddenSize') && params.hiddenSize !== d.hiddenSize) {
    vars.hiddenDims = [params.hiddenSize];
  }
  if (fields.has('epochs') && params.epochs !== d.epochs) vars.epochs = params.epochs;
  if (fields.has('patience') && params.patience !== d.patience) {
    vars.patience = params.patience;
  }

  if (params.seed !== d.seed) vars.seed = params.seed;
  if (params.trainSplit !== d.trainSplit) vars.trainSplit = params.trainSplit;
  if (params.maxTrainSamples !== d.maxTrainSamples) {
    vars.maxTrainSamples = params.maxTrainSamples;
  }

  return vars;
}
