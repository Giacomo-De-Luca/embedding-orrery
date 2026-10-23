'use client';

import { useState, useEffect, useMemo } from 'react';
import type { DataType, MediaOptionsInput, PortionStrategy } from '@/lib/graphql/mutations';
import { useEmbeddingModelCatalog } from '@/lib/hooks/useEmbeddingModelCatalog';
import { useMediaEmbeddingSettings } from '@/lib/hooks/useMediaEmbeddingSettings';
import type { ModelSuggestions } from '@/lib/utils/modelSuggestions';
import { useEmbeddingModelState, type EmbeddingModelState } from './useEmbeddingModelState';
import { modelOnDataTypeSwitch, saeLinkFromParams, updateTextTemplate, type CommonEmbedFormValues } from './embeddingFormUtils';
import type { MediaValidationInput } from './embedValidation';
import { MediaColumns, type MediaKind } from './mediaColumns';
import { MediaOptionsBudget } from './mediaOptions';
import { placeholderIssues as findPlaceholderIssues, type TaskPrefixRow } from './taskPrefixes';
import { ColumnDefaults, type ColumnDefaultsInput, type TextPick } from './columnDefaults';

/** Media columns of the loaded source and their kinds (MediaColumns.kindsOf). */
export type MediaColumnKinds = Record<string, MediaKind>;

export interface EmbedFormState {
  /** Model + topic configuration (shared hook) */
  model: EmbeddingModelState;
  /** The embed forms' model catalog (suggestions, the multimodal default, modalities) */
  modelCatalog: ModelSuggestions;
  /** The model an empty name runs in MULTIMODAL mode (the backend's default); null until known */
  multimodalDefaultModel: string | null;

  /** TEXT, MULTIMODAL (Image / Audio / Video / Mixed) or VECTOR (local files only) */
  dataType: DataType;
  /**
   * Switch the data type. MULTIMODAL needs a sentence-transformers media
   * model: the provider switches to it and the model name empties (the
   * backend's default) unless a media model is already typed.
   */
  setDataType: (v: DataType) => void;

  collectionName: string;
  setCollectionName: (v: string) => void;

  /** Text columns (MULTIMODAL: the optional text embedded with the media) */
  selectedEmbeddingColumns: string[];
  setSelectedEmbeddingColumns: (v: string[]) => void;
  selectedMetadataColumns: string[];
  setSelectedMetadataColumns: (v: string[]) => void;
  /** MULTIMODAL: the image / audio / video columns */
  selectedMediaColumns: string[];
  setSelectedMediaColumns: (v: string[]) => void;
  textTemplate: string;
  setTextTemplate: (v: string) => void;
  idColumn: string;
  setIdColumn: (v: string) => void;
  /** Column toggle that keeps the text template's placeholders in sync */
  handleEmbeddingColumnsChange: (cols: string[]) => void;
  /**
   * Start a loaded source from its default selection (ColumnDefaults): the
   * text column picked by name or by length, media columns, the id column.
   */
  applyColumnDefaults: (input: ColumnDefaultsInput) => void;
  /** Why the embedded column was pre-selected, while it still is the selection; null otherwise. */
  selectionNote: string | null;

  /** MULTIMODAL: the media options changed from their defaults */
  mediaOptions: MediaOptionsInput;
  setMediaOptions: (v: MediaOptionsInput) => void;
  /** Limits and token estimate of the media options; null until the settings load */
  mediaBudget: MediaOptionsBudget | null;
  /** Why the media settings failed to load (null otherwise); `retryMediaSettings` refetches them */
  mediaSettingsError: string | null;
  retryMediaSettings: () => void;

  portionStrategy: PortionStrategy;
  setPortionStrategy: (v: PortionStrategy) => void;
  numRows: number;
  setNumRows: (v: number) => void;
  rangeStart: number;
  setRangeStart: (v: number) => void;
  rangeEnd: number;
  setRangeEnd: (v: number) => void;
  randomSeed: number;
  setRandomSeed: (v: number) => void;

  /** Snapshot of the shared values consumed by the pure input builders */
  commonValues: (mediaKinds?: MediaColumnKinds) => CommonEmbedFormValues;
  /**
   * Text template / task prefix problems the backend would reject; the
   * prefix may name any of `prefixColumns` (the source's columns).
   */
  placeholderIssues: (prefixColumns: string[]) => string[];
  /** The row the task prefix is written for (EmbeddingModelForm), over the same `prefixColumns`. */
  taskPrefixRow: (prefixColumns: string[]) => TaskPrefixRow;
  /** What the MULTIMODAL checks of embedValidation need. */
  mediaValidation: (mediaKinds: MediaColumnKinds) => MediaValidationInput;
}

/**
 * The embed-form state shared verbatim between the HuggingFace and Local
 * Files tabs (each tab instantiates its own copy — state stays per-tab).
 * Column/template/id state resets when `resetKey` (dataset id or file path)
 * changes, matching the previous per-tab effects.
 */
export function useEmbedFormState(resetKey: string): EmbedFormState {
  const [dataType, setDataTypeState] = useState<DataType>('TEXT');
  const { suggestions: modelCatalog } = useEmbeddingModelCatalog();
  const {
    settings: mediaSettings,
    error: mediaSettingsError,
    refetch: retryMediaSettings,
  } = useMediaEmbeddingSettings();
  const mediaBudget = useMemo(() => (mediaSettings ? new MediaOptionsBudget(mediaSettings) : null), [mediaSettings]);
  const multimodalDefault =
    modelCatalog.defaultFor('multimodal')?.modelName ?? mediaSettings?.defaultModel ?? null;
  const model = useEmbeddingModelState({
    fallbackModelName: dataType === 'MULTIMODAL' ? multimodalDefault : null,
  });

  const [collectionName, setCollectionName] = useState('');

  // Column configuration
  const [selectedEmbeddingColumns, setSelectedEmbeddingColumns] = useState<string[]>([]);
  const [selectedMetadataColumns, setSelectedMetadataColumns] = useState<string[]>([]);
  const [selectedMediaColumns, setSelectedMediaColumns] = useState<string[]>([]);
  const [textTemplate, setTextTemplate] = useState('');
  const [idColumn, setIdColumn] = useState('auto');
  const [textPick, setTextPick] = useState<TextPick | null>(null);
  const [mediaOptions, setMediaOptions] = useState<MediaOptionsInput>({});

  // Portion configuration
  const [portionStrategy, setPortionStrategy] = useState<PortionStrategy>('FIRST_N');
  const [numRows, setNumRows] = useState(1000);
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(1000);
  const [randomSeed, setRandomSeed] = useState(42);

  const handleEmbeddingColumnsChange = (cols: string[]) => {
    setSelectedEmbeddingColumns(cols);
    setTextTemplate(updateTextTemplate(textTemplate, selectedEmbeddingColumns, cols));
    // A hand-picked selection needs no "Pre-selected" note.
    setTextPick(null);
  };

  const applyColumnDefaults = (input: ColumnDefaultsInput) => {
    const selection = ColumnDefaults.select(input);
    setSelectedMediaColumns(selection.media);
    setSelectedEmbeddingColumns(selection.embedding);
    setSelectedMetadataColumns(selection.metadata);
    setTextTemplate(selection.template);
    if (selection.idColumn) setIdColumn(selection.idColumn);
    setTextPick(selection.textPick);
  };

  const setDataType = (next: DataType) => {
    const switched = modelOnDataTypeSwitch({
      from: dataType,
      to: next,
      provider: model.embeddingProvider,
      modelName: model.modelName,
      catalog: modelCatalog,
    });
    if (switched) model.handleProviderChange(switched.provider, switched.modelName);
    setDataTypeState(next);
  };

  // Reset column config when the data source changes
  useEffect(() => {
    setSelectedEmbeddingColumns([]);
    setSelectedMetadataColumns([]);
    setSelectedMediaColumns([]);
    setTextTemplate('');
    setIdColumn('auto');
    setTextPick(null);
  }, [resetKey]);

  const commonValues = (mediaKinds: MediaColumnKinds = {}): CommonEmbedFormValues => ({
    collectionName,
    selectedEmbeddingColumns,
    selectedMetadataColumns,
    selectedMediaColumns,
    mediaOptions:
      dataType === 'MULTIMODAL' && mediaBudget
        ? mediaBudget.request(mediaOptions, MediaColumns.counts(selectedMediaColumns, mediaKinds))
        : undefined,
    textTemplate,
    idColumn,
    batchSize: model.batchSize,
    embeddingModel: model.buildEmbeddingModelInput(),
    topicParams: model.getTopicParams(),
    // SAE activations read text documents: media embeds link no SAE (the section is hidden).
    saeLink: dataType === 'MULTIMODAL' ? null : saeLinkFromParams(model.getSaePostParams()),
  });

  const placeholderIssues = (prefixColumns: string[]): string[] => findPlaceholderIssues({
    template: textTemplate,
    embeddingColumns: selectedEmbeddingColumns,
    prefix: model.taskPrefix?.documentPrefix,
    prefixColumns,
  });

  const taskPrefixRow = (prefixColumns: string[]): TaskPrefixRow => ({
    columns: prefixColumns,
    template: textTemplate,
    embeddingColumns: selectedEmbeddingColumns,
  });

  const mediaValidation = (mediaKinds: MediaColumnKinds): MediaValidationInput => {
    const hasText = MediaColumns.withoutMedia(selectedEmbeddingColumns, mediaKinds).length > 0;
    const wanted = [
      ...(hasText ? ['text'] : []),
      ...MediaColumns.selectedKinds(selectedMediaColumns, mediaKinds),
    ];
    return {
      columns: selectedMediaColumns,
      kinds: MediaColumns.counts(selectedMediaColumns, mediaKinds),
      budget: mediaBudget,
      settingsError: mediaSettingsError,
      overrides: mediaOptions,
      modelName: model.effectiveModelName,
      missingModalities: modelCatalog.missingModalities(model.embeddingProvider, model.effectiveModelName, wanted),
    };
  };

  return {
    model,
    modelCatalog,
    multimodalDefaultModel: multimodalDefault,
    dataType,
    setDataType,
    collectionName,
    setCollectionName,
    selectedEmbeddingColumns,
    setSelectedEmbeddingColumns,
    selectedMetadataColumns,
    setSelectedMetadataColumns,
    selectedMediaColumns,
    setSelectedMediaColumns,
    textTemplate,
    setTextTemplate,
    idColumn,
    setIdColumn,
    handleEmbeddingColumnsChange,
    applyColumnDefaults,
    selectionNote: ColumnDefaults.note(textPick, selectedEmbeddingColumns, dataType),
    mediaOptions,
    setMediaOptions,
    mediaBudget,
    mediaSettingsError,
    retryMediaSettings,
    portionStrategy,
    setPortionStrategy,
    numRows,
    setNumRows,
    rangeStart,
    setRangeStart,
    rangeEnd,
    setRangeEnd,
    randomSeed,
    setRandomSeed,
    commonValues,
    placeholderIssues,
    taskPrefixRow,
    mediaValidation,
  };
}
