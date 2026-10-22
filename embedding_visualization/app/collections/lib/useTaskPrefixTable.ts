'use client';

import { useQuery } from '@apollo/client/react';
import { EMBEDDING_TASK_PREFIXES } from '@/lib/graphql/queries';
import type { TaskPrefixTable } from './taskPrefixes';

interface TaskPrefixData {
  embeddingTaskPrefixes: TaskPrefixTable;
}

export interface TaskPrefixTableState {
  table: TaskPrefixTable | null;
  loading: boolean;
}

/**
 * The task-prefix table, fetched once per page (cache-first) so a model's
 * prefixes are known as soon as it is typed: no per-model lookup to race the
 * embed button. A failed fetch leaves `table` null (no model takes prefixes).
 */
export function useTaskPrefixTable(): TaskPrefixTableState {
  const { data, loading } = useQuery<TaskPrefixData>(EMBEDDING_TASK_PREFIXES, {
    fetchPolicy: 'cache-first',
  });
  return { table: data?.embeddingTaskPrefixes ?? null, loading };
}
