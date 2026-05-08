'use client';

import { useState } from 'react';
import { useMutation } from '@apollo/client/react';
import { Download } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Card, CardContent } from '@/lib/ui-primitives/card';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { ProgressModal } from './EmbeddingProgressModal';
import { PREPARE_SAE_DATA, type PrepareSaeResult } from '@/lib/graphql/mutations';

// ── Constants ───────────────────────────────────────────────────────────────

const LABELLED_LAYERS = new Set([9, 17, 22, 29]);

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

// ── Component ───────────────────────────────────────────────────────────────

export function SaeTab() {
  // Form state
  const [layer, setLayer] = useState(9);
  const [width, setWidth] = useState('16k');
  const [hookType, setHookType] = useState('resid_post');
  const [includeActivations, setIncludeActivations] = useState(false);

  // Job state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<PrepareSaeResult | null>(null);

  const [prepareSae, { loading: prepareLoading }] = useMutation<{
    prepareSaeData: PrepareSaeResult;
  }>(PREPARE_SAE_DATA, {
    onCompleted: (data) => {
      setLastResult(data.prepareSaeData);
      setActiveJobId(null);
    },
    onError: (err) => {
      setLastResult({
        modelId: '',
        saeId: '',
        featuresParquet: null,
        activationsJsonl: null,
        durationSeconds: 0,
        status: 'failed',
        error: err.message,
      });
      setActiveJobId(null);
    },
  });

  const handleDownload = () => {
    const jobId = `sae_prepare_${layer}_${hookType}_${width}`;
    setActiveJobId(jobId);
    setLastResult(null);
    prepareSae({
      variables: {
        input: {
          layer,
          width,
          hookType,
          includeActivations,
          skipDownload: false,
        },
      },
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6 space-y-4">
          <h3 className="text-sm font-semibold">Download SAE Data</h3>
          <p className="text-xs text-muted-foreground">
            Download features and decoder vectors from Neuronpedia S3. The output
            parquet can be imported as a vector collection via the Local Files tab.
          </p>

          <div className="flex flex-wrap items-end gap-3">
            {/* Layer */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Layer</label>
              <Select value={String(layer)} onValueChange={(v) => setLayer(Number(v))}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 34 }, (_, i) => (
                    <SelectItem key={i} value={String(i)}>
                      <span className="flex items-center gap-1.5">
                        {i}
                        {LABELLED_LAYERS.has(i) && (
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

          {/* Action */}
          <div className="flex items-center gap-3">
            <Button
              onClick={handleDownload}
              disabled={prepareLoading || !!activeJobId}
              size="sm"
            >
              <Download className="h-3.5 w-3.5 mr-1.5" />
              Download
            </Button>
          </div>

          {/* Result display */}
          {lastResult && <ResultDisplay result={lastResult} />}
        </CardContent>
      </Card>

      {/* Progress Modal */}
      {activeJobId && (
        <ProgressModal
          jobId={activeJobId}
          title="Downloading SAE Data"
          subtitle="Downloading from Neuronpedia and extracting decoder vectors. This may take several minutes."
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

  const isAlready = result.status === 'already_downloaded';

  return (
    <div className="rounded-md bg-muted/50 p-3 space-y-1">
      <div className="flex items-center gap-2">
        <Badge
          className={
            isAlready
              ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400 text-xs'
              : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 text-xs'
          }
        >
          {isAlready ? 'Already downloaded' : `Completed in ${result.durationSeconds.toFixed(1)}s`}
        </Badge>
        <span className="text-xs text-muted-foreground">
          {result.modelId} / {result.saeId}
        </span>
      </div>
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
      {result.featuresParquet && (
        <p className="text-xs text-muted-foreground mt-2">
          Import the parquet via the <strong>Local Files</strong> tab as a vector collection to visualize.
        </p>
      )}
    </div>
  );
}
