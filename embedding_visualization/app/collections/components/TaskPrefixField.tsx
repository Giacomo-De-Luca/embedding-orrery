'use client';

import { Input } from '@/lib/ui-primitives/input';
import { Label } from '@/lib/ui-primitives/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import {
  NO_TASK,
  modelInputPattern,
  placeholderIssues,
  queryPrefixOf,
  taskPrefixValue,
  type TaskPrefixInfo,
  type TaskPrefixValue,
} from '../lib/taskPrefixes';
import type { EmbeddingModelState } from '../lib/useEmbeddingModelState';

/** A model's task prefixes and the form's choice (from useEmbeddingModelState). */
export interface TaskPrefixControl {
  info: TaskPrefixInfo | null;
  value: TaskPrefixValue | null;
  onChange: (value: TaskPrefixValue) => void;
}

export function taskPrefixControlOf(
  model: Pick<EmbeddingModelState, 'taskPrefixInfo' | 'taskPrefix' | 'setTaskPrefix'>,
): TaskPrefixControl {
  return { info: model.taskPrefixInfo, value: model.taskPrefix, onChange: model.setTaskPrefix };
}

interface TaskPrefixFieldProps {
  info: TaskPrefixInfo;
  value: TaskPrefixValue;
  onChange: (value: TaskPrefixValue) => void;
  /** Columns a prefix placeholder may name ([] = static prefixes only). */
  columns: string[];
  /** The text template and embedded columns, for the model-input line. */
  template?: string;
  embeddingColumns?: string[];
  idPrefix?: string;
}

/**
 * Task picker + editable document prefix for models that read their task from
 * a prefix (EmbeddingGemma, Gemini Embedding 2). Picking a task fills in its
 * document prefix, which can then be edited (`title: {title} | text: `); the
 * query prefix follows the task.
 */
export function TaskPrefixField({
  info,
  value,
  onChange,
  columns,
  template,
  embeddingColumns,
  idPrefix = '',
}: TaskPrefixFieldProps) {
  const queryPrefix = queryPrefixOf(info, value);
  const noTask = value.taskId === NO_TASK;
  const issues = placeholderIssues({ prefix: value.documentPrefix, prefixColumns: columns });
  const pattern =
    embeddingColumns && embeddingColumns.length > 0
      ? modelInputPattern(value.documentPrefix, template ?? '', embeddingColumns)
      : null;

  return (
    <div className="space-y-2 rounded-md border p-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-[12rem_minmax(0,1fr)]">
        <div className="space-y-2">
          <Label htmlFor={`${idPrefix}task-prefix-task`}>Task</Label>
          <Select value={value.taskId} onValueChange={(id) => onChange(taskPrefixValue(info, id))}>
            <SelectTrigger id={`${idPrefix}task-prefix-task`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {info.tasks.map((task) => (
                <SelectItem key={task.id} value={task.id}>
                  {task.label}
                </SelectItem>
              ))}
              <SelectItem value={NO_TASK}>None</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label htmlFor={`${idPrefix}task-prefix-document`}>Document prefix</Label>
          <Input
            id={`${idPrefix}task-prefix-document`}
            value={value.documentPrefix}
            onChange={(e) => onChange({ taskId: value.taskId, documentPrefix: e.target.value })}
            disabled={noTask}
            placeholder={noTask ? 'No prefix' : 'title: {title} | text: '}
            className="font-mono text-sm"
            aria-invalid={issues.length > 0 ? true : undefined}
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Added before each document for the model only: the stored text stays unchanged.
        Use {'{column}'} for a column&apos;s value, e.g. <code className="font-mono">title: {'{title}'} | text: </code>
        (an empty value reads <code className="font-mono">none</code>).
      </p>
      <p className="text-xs text-muted-foreground">
        Queries use:{' '}
        {queryPrefix ? <code className="font-mono">{queryPrefix}…</code> : 'no prefix'}
      </p>
      {pattern && (
        <p className="text-xs text-muted-foreground break-all">
          Model input: <code className="font-mono">{pattern}</code>
        </p>
      )}
      {issues.map((issue) => (
        <p key={issue} className="text-xs text-destructive">
          {issue}
        </p>
      ))}
    </div>
  );
}
