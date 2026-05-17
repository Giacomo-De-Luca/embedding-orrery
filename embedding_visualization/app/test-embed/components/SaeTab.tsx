'use client';

import { useState, useEffect } from 'react';
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
import { GET_SAE_MODELS } from '@/lib/graphql/queries';
import {
  PREPARE_SAE_DATA,
  DELETE_SAE_DATA,
  type PrepareSaeResult,
  type SaeCollectionMode,
} from '@/lib/graphql/mutations';
import type { SaeModelInfo } from '@/lib/types/types';

// ── Constants ───────────────────────────────────────────────────────────────

interface GemmaModelConfig {
  size: string;
  label: string;
  layers: number;
  dIn: number;
  labelledLayers: Set<number>;
}

const GEMMA_MODELS: GemmaModelConfig[] = [
  { size: '1b', label: 'Gemma 3 1B', layers: 26, dIn: 1152, labelledLayers: new Set() },
  { size: '4b', label: 'Gemma 3 4B', layers: 34, dIn: 2560, labelledLayers: new Set([9, 17, 22, 29]) },
  { size: '12b', label: 'Gemma 3 12B', layers: 48, dIn: 3840, labelledLayers: new Set() },
  { size: '27b', label: 'Gemma 3 27B', layers: 62, dIn: 5376, labelledLayers: new Set() },
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
  const [modelSize, setModelSize] = useState('4b');
  const [variant, setVariant] = useState('it');
  const activeModel = GEMMA_MODELS.find((m) => m.size === modelSize) ?? GEMMA_MODELS[1];

  // Form state
  const [layer, setLayer] = useState(9);
  const [width, setWidth] = useState('16k');
  const [hookType, setHookType] = useState('resid_post');
  const [includeActivations, setIncludeActivations] = useState(false);

  // Clamp layer when model changes
  useEffect(() => {
    if (layer >= activeModel.layers) {
      setLayer(0);
    }
  }, [modelSize]); // eslint-disable-line react-hooks/exhaustive-deps
  // Collection creation state
  const [createCollection, setCreateCollection] = useState(false);
  const [collectionMode, setCollectionMode] = useState<SaeCollectionMode>('DECODER_VECTORS');
  const [extractTopics, setExtractTopics] = useState(false);
  const [deleteSourceFiles, setDeleteSourceFiles] = useState(false);

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
    const jobId = `sae_prepare_${modelSize}_${variant}_${layer}_${hookType}_${width}`;
    setActiveJobId(jobId);
    setLastResult(null);
    prepareSae({
      variables: {
        input: {
          layer,
          width,
          hookType,
          modelSize,
          variant,
          includeActivations,
          skipDownload: false,
          createCollection,
          collectionMode: createCollection ? collectionMode : undefined,
          extractTopics: createCollection ? extractTopics : undefined,
          deleteSourceFiles: createCollection ? deleteSourceFiles : undefined,
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
            Download features and decoder vectors from Neuronpedia S3. Features and
            activations are ingested into DuckDB. The output parquet can also be imported
            as a vector collection via the Local Files tab for visualization.
          </p>

          <div className="flex flex-wrap items-end gap-3">
            {/* Model Size */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Model</label>
              <Select value={modelSize} onValueChange={setModelSize}>
                <SelectTrigger className="w-36">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {GEMMA_MODELS.map((m) => (
                    <SelectItem key={m.size} value={m.size}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Variant */}
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

            {/* Width */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Width</label>
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
            </div>

            {/* Hook Type */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Hook Type</label>
              <Select value={hookType} onValueChange={setHookType}>
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

          {/* Activations checkbox with dynamic size */}
          <label className="flex items-center gap-2 text-xs">
            <Checkbox
              checked={includeActivations}
              onCheckedChange={(c) => setIncludeActivations(c === true)}
            />
            Include activation examples ({ACTIVATION_SIZE[width] ?? '~336 MB'})
          </label>

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
                  checked={extractTopics}
                  onCheckedChange={(c) => setExtractTopics(c === true)}
                />
                Extract topics (HDBSCAN + c-TF-IDF)
              </label>

              <label className="flex items-center gap-2 text-xs">
                <Checkbox
                  checked={deleteSourceFiles}
                  onCheckedChange={(c) => setDeleteSourceFiles(c === true)}
                />
                Delete source files after creation
              </label>
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
