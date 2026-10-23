'use client';

import { OptionalStepCard } from './CollapsibleCard';
import { TopicConfigForm } from './TopicConfigForm';
import type { EmbeddingModelState } from '../lib/useEmbeddingModelState';

interface TopicExtractionSectionProps {
  model: Pick<EmbeddingModelState, 'enableTopics' | 'setEnableTopics' | 'topicConfig' | 'setTopicConfig'>;
  /** Checkbox label (pre-computed vectors: "Extract topics after import"). */
  label?: string;
  idPrefix?: string;
}

/**
 * Optional step of the embed flows (HuggingFace, local file, re-embed):
 * extract topics once the collection is embedded. The settings travel with
 * the embed input through `model.getTopicParams()`.
 */
export function TopicExtractionSection({
  model,
  label = 'Extract topics after embedding',
  idPrefix = '',
}: TopicExtractionSectionProps) {
  return (
    <OptionalStepCard
      title="Topic Extraction"
      description="Optional: group the documents into labelled topics."
      checkboxId={`${idPrefix}enable-topics`}
      checkboxLabel={label}
      enabled={model.enableTopics}
      onEnabledChange={model.setEnableTopics}
    >
      <TopicConfigForm value={model.topicConfig} onChange={model.setTopicConfig} />
    </OptionalStepCard>
  );
}
