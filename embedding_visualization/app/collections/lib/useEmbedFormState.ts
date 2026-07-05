'use client';

import { useState, useEffect } from 'react';
import type { PortionStrategy } from '@/lib/graphql/mutations';
import { useEmbeddingModelState, type EmbeddingModelState } from './useEmbeddingModelState';
import { updateTextTemplate, type CommonEmbedFormValues } from './embeddingFormUtils';

export interface EmbedFormState {
  /** Model + topic configuration (shared hook) */
  model: EmbeddingModelState;

  collectionName: string;
  setCollectionName: (v: string) => void;

  selectedEmbeddingColumns: string[];
  setSelectedEmbeddingColumns: (v: string[]) => void;
  selectedMetadataColumns: string[];
  setSelectedMetadataColumns: (v: string[]) => void;
  textTemplate: string;
  setTextTemplate: (v: string) => void;
  idColumn: string;
  setIdColumn: (v: string) => void;
  /** Column toggle that keeps the text template's placeholders in sync */
  handleEmbeddingColumnsChange: (cols: string[]) => void;

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
  commonValues: () => CommonEmbedFormValues;
}

/**
 * The embed-form state shared verbatim between the HuggingFace and Local
 * Files tabs (each tab instantiates its own copy — state stays per-tab).
 * Column/template/id state resets when `resetKey` (dataset id or file path)
 * changes, matching the previous per-tab effects.
 */
export function useEmbedFormState(resetKey: string): EmbedFormState {
  const model = useEmbeddingModelState();

  const [collectionName, setCollectionName] = useState('');

  // Column configuration
  const [selectedEmbeddingColumns, setSelectedEmbeddingColumns] = useState<string[]>([]);
  const [selectedMetadataColumns, setSelectedMetadataColumns] = useState<string[]>([]);
  const [textTemplate, setTextTemplate] = useState('');
  const [idColumn, setIdColumn] = useState('auto');

  // Portion configuration
  const [portionStrategy, setPortionStrategy] = useState<PortionStrategy>('FIRST_N');
  const [numRows, setNumRows] = useState(1000);
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(1000);
  const [randomSeed, setRandomSeed] = useState(42);

  const handleEmbeddingColumnsChange = (cols: string[]) => {
    setSelectedEmbeddingColumns(cols);
    setTextTemplate(updateTextTemplate(textTemplate, selectedEmbeddingColumns, cols));
  };

  // Reset column config when the data source changes
  useEffect(() => {
    setSelectedEmbeddingColumns([]);
    setSelectedMetadataColumns([]);
    setTextTemplate('');
    setIdColumn('auto');
  }, [resetKey]);

  const commonValues = (): CommonEmbedFormValues => ({
    collectionName,
    selectedEmbeddingColumns,
    selectedMetadataColumns,
    textTemplate,
    idColumn,
    batchSize: model.batchSize,
    embeddingModel: model.buildEmbeddingModelInput(),
    topicParams: model.getTopicParams(),
  });

  return {
    model,
    collectionName,
    setCollectionName,
    selectedEmbeddingColumns,
    setSelectedEmbeddingColumns,
    selectedMetadataColumns,
    setSelectedMetadataColumns,
    textTemplate,
    setTextTemplate,
    idColumn,
    setIdColumn,
    handleEmbeddingColumnsChange,
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
  };
}
