'use client';

import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@apollo/client/react';
import { ChevronDown, Cpu } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/lib/ui-primitives/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/lib/ui-primitives/popover';
import { cn } from '@/lib/utils/utils';
import { MODEL_STATUS, GET_SAE_MODELS } from '@/lib/graphql/queries';
import { LOAD_MODEL, UNLOAD_MODEL } from '@/lib/graphql/mutations';
import { isModelMatch } from '@/lib/utils/modelLoader';
import { useModelIdentityStore } from '@/lib/stores/useModelIdentityStore';

interface SaeModelInfo {
  modelId: string;
  saeId: string;
  featureCount: number;
  activationCount: number;
}

interface ModelStatusButtonProps {
  onSelectModel?: (modelId: string, saeId: string) => void;
}

function PureModelStatusButton({ onSelectModel }: ModelStatusButtonProps) {
  const modelId = useModelIdentityStore((s) => s.modelId);
  const checkpoint = useModelIdentityStore((s) => s.checkpoint);
  const [transitioning, setTransitioning] = useState(false);
  const [open, setOpen] = useState(false);

  // Model status polling
  const { data: statusData, refetch } = useQuery<{
    modelStatus: {
      loaded: boolean;
      modelName: string | null;
      device: string | null;
      variant: string | null;
      modelSize: string | null;
    };
  }>(MODEL_STATUS, {
    pollInterval: 5000,
    fetchPolicy: 'network-only',
  });

  // Sync backend status into the store
  useEffect(() => {
    if (statusData?.modelStatus) {
      const ms = statusData.modelStatus;
      useModelIdentityStore.getState().syncBackendStatus({
        loaded: ms.loaded,
        modelName: ms.modelName,
        device: ms.device,
        variant: ms.variant,
        modelSize: ms.modelSize,
      });
    }
  }, [statusData]);

  // Available models list (reuse same query as SteeringSelector)
  const { data: modelsData } = useQuery<{ saeModels: SaeModelInfo[] }>(GET_SAE_MODELS);

  const [loadModel] = useMutation(LOAD_MODEL);
  const [unloadModel] = useMutation(UNLOAD_MODEL);

  // Derive status
  const loaded = statusData?.modelStatus?.loaded ?? false;
  const loadedModelName = statusData?.modelStatus?.modelName ?? null;
  const isCorrectModel = loaded && !!checkpoint && isModelMatch(loadedModelName, checkpoint);

  const dotColorClass = transitioning
    ? 'bg-amber-400 animate-pulse'
    : isCorrectModel
      ? 'bg-blue-500'
      : 'bg-red-500';

  // Unique model names with their first SAE as default
  const availableModels = useMemo(() => {
    if (!modelsData?.saeModels) return [];
    const seen = new Map<string, SaeModelInfo>();
    for (const m of modelsData.saeModels) {
      if (!seen.has(m.modelId)) seen.set(m.modelId, m);
    }
    return [...seen.values()];
  }, [modelsData?.saeModels]);

  // Toggle load/unload
  const toggleModel = useCallback(async () => {
    if (!checkpoint || transitioning) return;
    setTransitioning(true);
    try {
      if (isCorrectModel) {
        await unloadModel();
      } else {
        if (loaded) await unloadModel();
        await loadModel({ variables: { checkpoint } });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Model operation failed');
    } finally {
      await refetch();
      setTransitioning(false);
    }
  }, [checkpoint, transitioning, isCorrectModel, loaded, loadModel, unloadModel, refetch]);

  // Handle model selection from the dropdown
  const handleSelect = useCallback((mId: string, sId: string) => {
    onSelectModel?.(mId, sId);
    setOpen(false);
  }, [onSelectModel]);

  const displayLabel = modelId ?? 'No model';
  const backendVariant = statusData?.modelStatus?.variant ?? null;
  const variantLabel = isCorrectModel && backendVariant
    ? backendVariant === 'pt' ? 'Base' : 'Chat'
    : null;

  return (
    <div className="flex items-center gap-0.5">
      {/* Status dot — separate button for load/unload */}
      <button
        type="button"
        onClick={toggleModel}
        disabled={transitioning || !modelId}
        className="flex h-7 w-5 items-center justify-center rounded-md transition-colors hover:bg-muted"
        title={
          transitioning
            ? 'Loading...'
            : isCorrectModel
              ? `Model loaded (${variantLabel ?? 'unknown'}) — click to unload`
              : 'Model not loaded — click to load'
        }
      >
        <span className={cn('h-2 w-2 rounded-full transition-colors', dotColorClass)} />
      </button>

      {/* Model selector popover */}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            className="h-7 max-w-[210px] justify-start gap-1 rounded-lg px-1.5 text-[12px] text-muted-foreground transition-colors hover:text-foreground"
            variant="ghost"
          >
            <span className="truncate">{displayLabel}</span>
            {variantLabel && (
              <span className={cn(
                'shrink-0 rounded px-1 py-px text-[9px] font-medium leading-none',
                backendVariant === 'pt'
                  ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400'
                  : 'bg-blue-500/15 text-blue-600 dark:text-blue-400',
              )}>
                {variantLabel}
              </span>
            )}
            <ChevronDown className="ml-auto h-3 w-3 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>

      <PopoverContent
        side="top"
        sideOffset={8}
        align="start"
        className="w-64 p-0"
      >
        <div className="max-h-64 overflow-y-auto p-1">
          {availableModels.length === 0 ? (
            <div className="px-3 py-4 text-center text-xs text-muted-foreground">
              No models available
            </div>
          ) : (
            availableModels.map((item) => {
              const isSelected = item.modelId === modelId;
              return (
                <button
                  key={item.modelId}
                  type="button"
                  onClick={() => handleSelect(item.modelId, item.saeId)}
                  className={cn(
                    'flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs transition-colors',
                    'hover:bg-accent hover:text-accent-foreground',
                    isSelected && 'bg-accent/50 font-medium',
                  )}
                >
                  <Cpu className="h-3 w-3 shrink-0 text-muted-foreground" />
                  <span className="truncate">{item.modelId}</span>
                  {isSelected && (
                    <span className={cn('ml-auto h-1.5 w-1.5 shrink-0 rounded-full', dotColorClass)} />
                  )}
                </button>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
    </div>
  );
}

export const ModelStatusButton = memo(PureModelStatusButton);
