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
  type TaskPrefixRow,
  type TaskPrefixValue,
} from '../lib/taskPrefixes';

interface TaskPrefixFieldProps {
  info: TaskPrefixInfo;
  value: TaskPrefixValue;
  onChange: (value: TaskPrefixValue) => void;
  /** The columns the prefix may name, and the text it goes before (model-input preview). */
  row: TaskPrefixRow;
  idPrefix?: string;
}

/**
 * Task picker + editable document prefix for models that read their task from
 * a prefix (EmbeddingGemma, Gemini Embedding 2). Picking a task fills in its
 * document prefix, which can then be edited (`title: {title} | text: `); the
 * query prefix follows the task.
 */
export function TaskPrefixField({ info, value, onChange, row, idPrefix = '' }: TaskPrefixFieldProps) {
  const queryPrefix = queryPrefixOf(info, value);
  const noTask = value.taskId === NO_TASK;
  const issues = placeholderIssues({ prefix: value.documentPrefix, prefixColumns: row.columns });
  const pattern =
    row.embeddingColumns && row.embeddingColumns.length > 0
      ? modelInputPattern(value.documentPrefix, row.template ?? '', row.embeddingColumns)
      : null;

  return (
    <div className="space-y-2">
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
            // Empty sends no document prefix (queries keep the task's query prefix).
            placeholder="No prefix"
            className="font-mono text-sm"
            aria-invalid={issues.length > 0 ? true : undefined}
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Added before each document for the model only: the stored text stays unchanged.
        {row.columns.length > 0 && (
          <>
            {' '}<code className="font-mono">{'{column}'}</code> inserts a column&apos;s value, e.g.{' '}
            <code className="font-mono">title: {'{title}'} | text: </code> (an empty value reads{' '}
            <code className="font-mono">none</code>).
          </>
        )}
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
