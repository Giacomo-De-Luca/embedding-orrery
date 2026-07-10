'use client';

import { useState } from 'react';
import { useQuery, useMutation } from '@apollo/client/react';
import { Download, RefreshCw, Trash2, Check, X } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Card, CardContent } from '@/lib/ui-primitives/card';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Separator } from '@/lib/ui-primitives/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { ProgressModal } from './EmbeddingProgressModal';
import { EmbeddingModelForm } from './EmbeddingModelForm';
import { TopicConfigForm, toTopicConfigInput } from './TopicConfigForm';
import { useEmbeddingModelState } from '../lib/useEmbeddingModelState';
import { GET_SAE_MODELS } from '@/lib/graphql/queries';
import {
  PREPARE_SAE_DATA,
  DELETE_SAE_DATA,
  type PrepareSaeResult,
  type SaeCollectionMode,
} from '@/lib/graphql/mutations';
import type { SaeModelInfo } from '@/lib/types/types';

// ── Constants ───────────────────────────────────────────────────────────────

interface SaeModelConfig {
  /** Unique select value ("gemma-27b" vs "qwen-27B" would collide on size alone) */
  key: string;
  family: 'gemma' | 'qwen';
  /** Backend model_size ("4b" for gemma, "1.7B" for qwen) */
  size: string;
  label: string;
  layers: number;
  dIn: number;
  /** Qwen-scope ships exactly one width per model; gemma offers WIDTH_OPTIONS */
  fixedWidth?: string;
  labelledLayers: Set<number>;
}

// `labelledLayers` = published GemmaScope-2 residual-stream SAE layers on
// Neuronpedia (identical for pt and it variants). These layers carry both decoder
// vectors and explanation labels, so they gate the "labelled" badge in the Layer
// dropdown and enable "Label embeddings" vector mode.
// Source (retrieved 2026-07) — Neuronpedia datasets S3 catalog:
//   https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/?prefix=v1/<model_id>/&delimiter=/
// Qwen-scope TopK SAEs (residual-stream only, k pinned to 50 server-side) are
// not on Neuronpedia: decoder vectors are extracted straight from the HF
// weights with empty labels/densities and no activation examples, so
// labelledLayers stays empty. Sizes/layers/widths mirror QWEN_SCOPE_MODELS in
// interpret/sae/sae_config.py.
const SAE_MODELS: SaeModelConfig[] = [
  { key: 'gemma-1b', family: 'gemma', size: '1b', label: 'Gemma 3 1B', layers: 26, dIn: 1152, labelledLayers: new Set([7, 13, 17, 22]) },
  { key: 'gemma-4b', family: 'gemma', size: '4b', label: 'Gemma 3 4B', layers: 34, dIn: 2560, labelledLayers: new Set([9, 17, 22, 29]) },
  { key: 'gemma-12b', family: 'gemma', size: '12b', label: 'Gemma 3 12B', layers: 48, dIn: 3840, labelledLayers: new Set([12, 24, 31, 41]) },
  { key: 'gemma-27b', family: 'gemma', size: '27b', label: 'Gemma 3 27B', layers: 62, dIn: 5376, labelledLayers: new Set([16, 31, 40, 53]) },
  { key: 'qwen-1.7B', family: 'qwen', size: '1.7B', label: 'Qwen3 1.7B', layers: 28, dIn: 2048, fixedWidth: '32k', labelledLayers: new Set() },
  { key: 'qwen-2B', family: 'qwen', size: '2B', label: 'Qwen3.5 2B', layers: 24, dIn: 2048, fixedWidth: '32k', labelledLayers: new Set() },
  { key: 'qwen-8B', family: 'qwen', size: '8B', label: 'Qwen3 8B', layers: 36, dIn: 4096, fixedWidth: '64k', labelledLayers: new Set() },
  { key: 'qwen-27B', family: 'qwen', size: '27B', label: 'Qwen3.5 27B', layers: 64, dIn: 5120, fixedWidth: '80k', labelledLayers: new Set() },
];

const VARIANT_OPTIONS = [
  { value: 'it', label: 'Instruction-tuned (IT)' },
  { value: 'pt', label: 'Pretrained (PT)' },
];

const WIDTH_OPTIONS = [
  { value: '16k', label: '16k', desc: '16,384 features, ~160 MB' },
  { value: '65k', label: '65k', desc: '65,536 features, ~650 MB' },
  { value: '262k', label: '262k', desc: '262,144 features, ~2.6 GB' },
];

const HOOK_OPTIONS = [
  { value: 'resid_post', label: 'Residual Stream' },
  { value: 'mlp_out', label: 'MLP Output' },
  { value: 'attn_out', label: 'Attention Output' },
];

/** Approximate activation download size per width */
const ACTIVATION_SIZE: Record<string, string> = {
  '16k': '~336 MB',
  '65k': '~1.3 GB',
  '262k': '~5.2 GB',
};

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

// ── Component ───────────────────────────────────────────────────────────────

export function SaeTab() {
  // Model selection state
  const [modelKey, setModelKey] = useState('gemma-4b');
  const [variant, setVariant] = useState('it');
  const activeModel = SAE_MODELS.find((m) => m.key === modelKey) ?? SAE_MODELS[1];
  const isQwen = activeModel.family === 'qwen';

  // Form state
  const [layer, setLayer] = useState(9);
  const [width, setWidth] = useState('16k');
  const [hookType, setHookType] = useState('resid_post');
  const [includeActivations, setIncludeActivations] = useState(false);

  const handleModelChange = (key: string) => {
    setModelKey(key);
    const newModel = SAE_MODELS.find((m) => m.key === key);
    if (newModel && layer >= newModel.layers) {
      setLayer(0);
    }
  };

  // Collection creation state
  const [createCollection, setCreateCollection] = useState(false);
  const [collectionMode, setCollectionMode] = useState<SaeCollectionMode>('DECODER_VECTORS');
  const [deleteSourceFiles, setDeleteSourceFiles] = useState(false);

  // Embedding model & topic config (reuse shared hook + components)
  const embeddingModel = useEmbeddingModelState();
  // Label mode only takes effect on layers that actually have labels
  const isLabelMode =
    collectionMode === 'LABEL_EMBEDDINGS' && activeModel.labelledLayers.has(layer);

  // Job state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<PrepareSaeResult | null>(null);

  // Delete state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // SAE models query
  const {
    data: modelsData,
    loading: modelsLoading,
    refetch: refetchModels,
  } = useQuery<{ saeModels: SaeModelInfo[] }>(GET_SAE_MODELS);
  const models = modelsData?.saeModels ?? [];

  // Mutations
  const [prepareSae, { loading: prepareLoading }] = useMutation<{
    prepareSaeData: PrepareSaeResult;
  }>(PREPARE_SAE_DATA, {
    onCompleted: (data) => {
      setLastResult(data.prepareSaeData);
      setActiveJobId(null);
      refetchModels();
    },
    onError: (err) => {
      setLastResult({
        modelId: '',
        saeId: '',
        featuresParquet: null,
        activationsJsonl: null,
        featuresInserted: 0,
        activationsInserted: 0,
        durationSeconds: 0,
        status: 'failed',
        error: err.message,
        collectionName: null,
        collectionItems: 0,
      });
      setActiveJobId(null);
    },
  });

  const [deleteSae] = useMutation(DELETE_SAE_DATA, {
    onCompleted: () => {
      setDeleteTarget(null);
      refetchModels();
    },
    onError: () => {
      setDeleteTarget(null);
    },
  });

  const handleDownload = () => {
    // Qwen-scope is residual-stream only, single-width, unlabelled, no
    // activation examples — pin the fields the hidden controls would set.
    const effWidth = activeModel.fixedWidth ?? width;
    const effHook = isQwen ? 'resid_post' : hookType;
    const effVariant = isQwen ? 'base' : variant;
    // Label-embeddings mode needs labels; fall back when the selected layer
    // has none (mode picked on a labelled layer, then layer/model switched).
    const effMode: SaeCollectionMode = activeModel.labelledLayers.has(layer)
      ? collectionMode
      : 'DECODER_VECTORS';

    const jobId = `sae_prepare_${activeModel.size}_${effVariant}_${layer}_${effHook}_${effWidth}`;
    setActiveJobId(jobId);
    setLastResult(null);

    const topicParams = embeddingModel.getTopicParams();
    prepareSae({
      variables: {
        input: {
          layer,
          width: effWidth,
          hookType: effHook,
          modelSize: activeModel.size,
          variant: effVariant,
          family: activeModel.family,
          includeActivations: isQwen ? false : includeActivations,
          skipDownload: false,
          createCollection,
          ...(createCollection && {
            collectionMode: effMode,
            embeddingModel:
              effMode === 'LABEL_EMBEDDINGS'
                ? embeddingModel.buildEmbeddingModelInput()
                : undefined,
            extractTopics: topicParams.extractTopics,
            topicConfig: topicParams.topicConfig,
            deleteSourceFiles,
          }),
        },
      },
    });
  };

  const handleDelete = (modelId: string, saeId: string) => {
    deleteSae({ variables: { modelId, saeId } });
  };

  return (
    <div className="space-y-6">
      {/* Download form */}
      <Card>
        <CardContent className="pt-6 space-y-4">
          <h3 className="text-sm font-semibold">Download SAE Data</h3>
          <p className="text-xs text-muted-foreground">
            {isQwen ? (
              <>
                Extract decoder vectors from the Qwen-scope SAE weights on HuggingFace.
                Qwen-scope is not on Neuronpedia, so features are ingested without labels,
                densities, or activation examples (labels arrive with a later autointerp
                pass). The output parquet can also be imported as a vector collection via
                the Local Files tab for visualization.
              </>
            ) : (
              <>
                Download features and decoder vectors from Neuronpedia S3. Features and
                activations are ingested into DuckDB. The output parquet can also be
                imported as a vector collection via the Local Files tab for visualization.
              </>
            )}
          </p>

          <div className="flex flex-wrap items-end gap-3">
            {/* Model */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Model</label>
              <Select value={modelKey} onValueChange={handleModelChange}>
                <SelectTrigger className="w-36">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SAE_MODELS.map((m) => (
                    <SelectItem key={m.key} value={m.key}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Variant (gemma only — qwen-scope repos are fixed per model) */}
            {!isQwen && (
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">Variant</label>
                <Select value={variant} onValueChange={setVariant}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VARIANT_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-end gap-3">
            {/* Layer */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Layer</label>
              <Select value={String(layer)} onValueChange={(v) => setLayer(Number(v))}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: activeModel.layers }, (_, i) => (
                    <SelectItem key={i} value={String(i)}>
                      <span className="flex items-center gap-1.5">
                        {i}
                        {activeModel.labelledLayers.has(i) && (
                          <Badge variant="secondary" className="text-[10px] px-1 py-0">
                            labelled
                          </Badge>
                        )}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Width (fixed per model for qwen-scope) */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Width</label>
              {activeModel.fixedWidth ? (
                <Select value={activeModel.fixedWidth} disabled>
                  <SelectTrigger className="w-52">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={activeModel.fixedWidth}>
                      {activeModel.fixedWidth}
                    </SelectItem>
                  </SelectContent>
                </Select>
              ) : (
                <Select value={width} onValueChange={setWidth}>
                  <SelectTrigger className="w-52">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {WIDTH_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <span>{opt.label}</span>
                        <span className="ml-2 text-muted-foreground text-xs">{opt.desc}</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>

            {/* Hook Type (qwen-scope is residual-stream only) */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Hook Type</label>
              <Select
                value={isQwen ? 'resid_post' : hookType}
                onValueChange={setHookType}
                disabled={isQwen}
              >
                <SelectTrigger className="w-44">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {HOOK_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Activations checkbox with dynamic size (no activation data for qwen-scope) */}
          {!isQwen && (
            <label className="flex items-center gap-2 text-xs">
              <Checkbox
                checked={includeActivations}
                onCheckedChange={(c) => setIncludeActivations(c === true)}
              />
              Include activation examples ({ACTIVATION_SIZE[width] ?? '~336 MB'})
            </label>
          )}

          <Separator className="my-2" />

          {/* Collection creation options */}
          <label className="flex items-center gap-2 text-xs font-medium">
            <Checkbox
              checked={createCollection}
              onCheckedChange={(c) => setCreateCollection(c === true)}
            />
            Create visualization collection
          </label>

          {createCollection && (
            <div className="space-y-4">
              <div className="ml-6 space-y-2.5">
                <div className="space-y-1.5">
                  <label className="text-xs text-muted-foreground">Vector mode</label>
                  <Select
                    value={collectionMode}
                    onValueChange={(v) => setCollectionMode(v as SaeCollectionMode)}
                  >
                    <SelectTrigger className="w-64">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="DECODER_VECTORS">
                        Decoder vectors (geometric)
                      </SelectItem>
                      <SelectItem
                        value="LABEL_EMBEDDINGS"
                        disabled={!activeModel.labelledLayers.has(layer)}
                      >
                        Label embeddings (semantic)
                        {!activeModel.labelledLayers.has(layer) && (
                          <span className="ml-1 text-muted-foreground">— no labels</span>
                        )}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <label className="flex items-center gap-2 text-xs">
                  <Checkbox
                    checked={deleteSourceFiles}
                    onCheckedChange={(c) => setDeleteSourceFiles(c === true)}
                  />
                  Delete source files after creation
                </label>
              </div>

              {/* Embedding model (only for label embeddings mode) */}
              {isLabelMode && (
                <EmbeddingModelForm
                  model={embeddingModel}
                  showTopics={false}
                  idPrefix="sae-"
                />
              )}

              {/* Topic extraction config */}
              <Card>
                <CardContent className="pt-4 space-y-3">
                  <label className="flex items-center gap-2 text-xs font-medium">
                    <Checkbox
                      checked={embeddingModel.enableTopics}
                      onCheckedChange={(c) => embeddingModel.setEnableTopics(c === true)}
                    />
                    Extract topics
                  </label>
                  {embeddingModel.enableTopics && (
                    <TopicConfigForm
                      value={embeddingModel.topicConfig}
                      onChange={embeddingModel.setTopicConfig}
                    />
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Action */}
          <div className="flex items-center gap-3">
            <Button
              onClick={handleDownload}
              disabled={prepareLoading || !!activeJobId}
              size="sm"
            >
              <Download className="h-3.5 w-3.5 mr-1.5" />
              Download & Ingest
            </Button>
          </div>

          {/* Result display */}
          {lastResult && <ResultDisplay result={lastResult} />}
        </CardContent>
      </Card>

      <Separator />

      {/* Ingested models table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Ingested SAE Models</h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refetchModels()}
            disabled={modelsLoading}
          >
            <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${modelsLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {models.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4 text-center">
            No SAE models ingested yet.
          </p>
        ) : (
          <div className="border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">Model</th>
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">SAE ID</th>
                  <th className="text-right px-3 py-2 font-medium text-muted-foreground">Features</th>
                  <th className="text-right px-3 py-2 font-medium text-muted-foreground">Activations</th>
                  <th className="text-right px-3 py-2 font-medium text-muted-foreground w-24"></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const key = `${m.modelId}::${m.saeId}`;
                  const isDeleting = deleteTarget === key;

                  return (
                    <tr key={key} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 font-mono text-xs">{m.modelId}</td>
                      <td className="px-3 py-2 font-mono text-xs">{m.saeId}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{formatCount(m.featureCount)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{formatCount(m.activationCount)}</td>
                      <td className="px-3 py-2 text-right">
                        {isDeleting ? (
                          <span className="flex items-center justify-end gap-1">
                            <span className="text-xs text-muted-foreground mr-1">Delete?</span>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-destructive"
                              onClick={() => handleDelete(m.modelId, m.saeId)}
                            >
                              <Check className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => setDeleteTarget(null)}
                            >
                              <X className="h-3.5 w-3.5" />
                            </Button>
                          </span>
                        ) : (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-muted-foreground hover:text-destructive"
                            onClick={() => setDeleteTarget(key)}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Progress Modal */}
      {activeJobId && (
        <ProgressModal
          jobId={activeJobId}
          title={createCollection ? "Preparing SAE Collection" : "Downloading SAE Data"}
          subtitle={
            createCollection
              ? "Downloading, extracting, embedding, and computing projections. This may take several minutes."
              : isQwen
                ? "Downloading SAE weights from HuggingFace and extracting decoder vectors. This may take several minutes."
                : "Downloading from Neuronpedia and extracting decoder vectors. This may take several minutes."
          }
          itemsLabel="batches"
        />
      )}
    </div>
  );
}

// ── Result Display ──────────────────────────────────────────────────────────

function ResultDisplay({ result }: { result: PrepareSaeResult }) {
  if (result.status === 'failed') {
    return (
      <div className="rounded-md bg-destructive/10 p-3 text-sm">
        <Badge variant="destructive" className="text-xs mb-1">Failed</Badge>
        <p className="text-xs text-muted-foreground">{result.error}</p>
      </div>
    );
  }

  return (
    <div className="rounded-md bg-muted/50 p-3 space-y-1">
      <div className="flex items-center gap-2">
        <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 text-xs">
          Completed in {result.durationSeconds.toFixed(1)}s
        </Badge>
        <span className="text-xs text-muted-foreground">
          {result.modelId} / {result.saeId}
        </span>
      </div>
      {(result.featuresInserted > 0 || result.activationsInserted > 0) && (
        <p className="text-xs text-muted-foreground">
          Ingested: {result.featuresInserted.toLocaleString()} features
          {result.activationsInserted > 0 && `, ${result.activationsInserted.toLocaleString()} activations`}
          {' '}into DuckDB
        </p>
      )}
      {result.featuresParquet && (
        <p className="text-xs font-mono text-muted-foreground truncate">
          Parquet: {result.featuresParquet}
        </p>
      )}
      {result.activationsJsonl && (
        <p className="text-xs font-mono text-muted-foreground truncate">
          Activations: {result.activationsJsonl}
        </p>
      )}
      {result.collectionName ? (
        <p className="text-xs text-muted-foreground mt-2">
          Collection <strong>{result.collectionName}</strong> created with{' '}
          {result.collectionItems.toLocaleString()} items. View it on the{' '}
          <a href="/" className="underline text-primary">visualization page</a>.
        </p>
      ) : result.featuresParquet ? (
        <p className="text-xs text-muted-foreground mt-2">
          Import the parquet via the <strong>Local Files</strong> tab as a vector collection to visualize.
        </p>
      ) : null}
    </div>
  );
}
