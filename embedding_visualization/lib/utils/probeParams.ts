/**
 * Probe hyperparameter model + GraphQL input builder.
 *
 * Pure, UI-agnostic: the ProbeSection form holds a single ProbeParams object,
 * the settings popover edits the fields relevant to the active kind, and
 * `buildTrainProbeInput` turns (collection, field, kind, params) into the
 * `TrainProbeInput` variables — omitting anything left at its default so the
 * backend applies its own default (one source of truth).
 */

/** Kinds offered in the kind select (the MLP's Classifier switch picks its trained kind). */
export type ProbeKind =
  | 'ridge'
  | 'lasso'
  | 'massmean'
  | 'massmean_cov'
  | 'svr'
  | 'logreg'
  | 'svc'
  | 'mlp';

/** Kinds the backend trains and stores: the select kinds plus the MLP classifier. */
export type TrainedProbeKind = ProbeKind | 'mlp_clf';

export type MlpActivation = 'relu' | 'gelu' | 'tanh' | 'silu';

export const MLP_ACTIVATION_OPTIONS: MlpActivation[] = ['relu', 'gelu', 'tanh', 'silu'];

export interface ProbeParams {
  alpha: number; // ridge / lasso L2/L1
  c: number; // svr / logreg / svc inverse-regularisation
  kernel: 'rbf' | 'linear'; // svr / svc
  classWeight: 'none' | 'balanced'; // classifiers
  hiddenSize: number; // mlp single hidden layer width
  epochs: number; // mlp
  activation: MlpActivation; // mlp hidden-layer nonlinearity
  patience: number; // mlp early-stopping patience (dev-set based)
  devSplit: number; // mlp: train fraction held out for early stopping
  devSplitEnabled: boolean; // mlp: unchecked -> devSplit 0 sent (early stopping off)
  seed: number; // shared
  trainSplit: number; // shared (train fraction)
  minClassPct: number; // classifiers: min class size, % of labelled items
  mlpClassifier: boolean; // mlp: train as a classifier (mlp_clf) where the target allows
}

export const DEFAULT_PROBE_PARAMS: ProbeParams = {
  alpha: 1.0,
  c: 1.0,
  kernel: 'rbf',
  classWeight: 'none',
  hiddenSize: 256,
  epochs: 100,
  activation: 'relu',
  patience: 10,
  devSplit: 0.2,
  devSplitEnabled: true,
  seed: 7, // mirrors the backend ProbeConfig default — displayed = trained
  trainSplit: 0.8,
  minClassPct: 1, // mirrors the backend's DEFAULT_MIN_CLASS_FRACTION (0.01)
  mlpClassifier: false,
};

export const PROBE_KIND_OPTIONS: { value: ProbeKind; label: string }[] = [
  { value: 'ridge', label: 'Ridge (linear)' },
  { value: 'lasso', label: 'Lasso (sparse linear)' },
  { value: 'massmean', label: 'Mass-mean' },
  { value: 'massmean_cov', label: 'Mass-mean (cov-corrected)' },
  { value: 'svr', label: 'SVR (RBF)' },
  { value: 'logreg', label: 'Logistic' },
  { value: 'svc', label: 'SVC (RBF)' },
  { value: 'mlp', label: 'MLP (nonlinear)' },
];

/** Classifier kinds: binary or multiclass targets, probability scores. */
export const CLASSIFICATION_KINDS: readonly TrainedProbeKind[] = ['logreg', 'svc', 'mlp_clf'];

export function isClassificationKind(kind: string): boolean {
  return (CLASSIFICATION_KINDS as readonly string[]).includes(kind);
}

/** Short label for a stored probe kind (probe rows, color field names). */
export function probeKindLabel(kind: string): string {
  return kind === 'mlp_clf' ? 'mlp classifier' : kind;
}

type ProbeParamField =
  | 'alpha'
  | 'c'
  | 'kernel'
  | 'classWeight'
  | 'hiddenSize'
  | 'epochs'
  | 'activation'
  | 'devSplit'
  | 'patience'
  | 'minClassPct';

const MLP_PARAM_FIELDS: ProbeParamField[] = [
  'hiddenSize',
  'epochs',
  'activation',
  'devSplit',
  'patience',
];

/** Which parameter controls the settings popover should show for a kind. */
export function probeParamFields(kind: TrainedProbeKind): ProbeParamField[] {
  switch (kind) {
    case 'ridge':
    case 'lasso':
      return ['alpha'];
    case 'svr':
      return ['c', 'kernel'];
    case 'logreg':
      return ['c', 'classWeight', 'minClassPct'];
    case 'svc':
      return ['c', 'kernel', 'classWeight', 'minClassPct'];
    case 'mlp':
      return MLP_PARAM_FIELDS;
    case 'mlp_clf':
      return [...MLP_PARAM_FIELDS, 'classWeight', 'minClassPct'];
    case 'massmean':
    case 'massmean_cov':
    default:
      return [];
  }
}

export interface TrainProbeInputVars {
  collectionName: string;
  targetField: string;
  kind: TrainedProbeKind;
  alpha?: number;
  c?: number;
  kernel?: string;
  classWeight?: string;
  hiddenDims?: number[];
  epochs?: number;
  activation?: string;
  patience?: number;
  devSplit?: number;
  seed?: number;
  trainSplit?: number;
  minClassFraction?: number;
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
  kind: TrainedProbeKind,
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
  if (fields.has('activation') && params.activation !== d.activation) {
    vars.activation = params.activation;
  }
  if (fields.has('devSplit')) {
    if (!params.devSplitEnabled) {
      // Unchecked -> early stopping off (backend semantic: devSplit 0).
      vars.devSplit = 0;
    } else if (params.devSplit !== d.devSplit) {
      vars.devSplit = params.devSplit;
    }
  }
  // Patience only matters while early stopping is on.
  if (fields.has('patience') && params.devSplitEnabled && params.patience !== d.patience) {
    vars.patience = params.patience;
  }
  if (fields.has('minClassPct') && params.minClassPct !== d.minClassPct) {
    vars.minClassFraction = params.minClassPct / 100;
  }

  if (params.seed !== d.seed) vars.seed = params.seed;
  if (params.trainSplit !== d.trainSplit) vars.trainSplit = params.trainSplit;

  return vars;
}
