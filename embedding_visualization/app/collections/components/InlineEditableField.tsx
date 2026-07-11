'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Input } from '@/lib/ui-primitives/input';
import { Button } from '@/lib/ui-primitives/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Pencil, Lock, Check, X, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils/utils';

export interface SelectOption {
  value: string;
  label: string;
}

export interface InlineEditableFieldProps {
  /** The metadata field key */
  fieldKey: string;
  /** Display label for the field */
  label: string;
  /** Current value */
  value: unknown;
  /** Input type - text or select dropdown */
  type: 'text' | 'select';
  /** Options for select type */
  selectOptions?: SelectOption[];
  /** Whether this field is read-only */
  readOnly?: boolean;
  /** External saving state */
  isSaving?: boolean;
  /** External error message */
  error?: string | null;
  /** Show delete button for custom fields */
  showDeleteButton?: boolean;
  /** Called when user saves the field */
  onSave: (key: string, value: unknown) => Promise<boolean>;
  /** Called when user deletes the field */
  onDelete?: (key: string) => Promise<boolean>;
}

export function InlineEditableField({
  fieldKey,
  label,
  value,
  type,
  selectOptions = [],
  readOnly = false,
  isSaving = false,
  error,
  showDeleteButton = false,
  onSave,
  onDelete,
}: InlineEditableFieldProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const blurTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Format value for display
  const displayValue = value === null || value === undefined
    ? ''
    : typeof value === 'object'
      ? JSON.stringify(value)
      : String(value);

  // Initialize edit value when entering edit mode
  const handleStartEdit = useCallback(() => {
    if (readOnly || isSaving) return;
    setEditValue(displayValue);
    setIsEditing(true);
  }, [readOnly, isSaving, displayValue]);

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (blurTimeoutRef.current) {
        clearTimeout(blurTimeoutRef.current);
      }
    };
  }, []);

  const handleSave = useCallback(async () => {
    if (blurTimeoutRef.current) {
      clearTimeout(blurTimeoutRef.current);
      blurTimeoutRef.current = null;
    }

    // Don't save if value hasn't changed
    if (editValue === displayValue) {
      setIsEditing(false);
      return;
    }

    const success = await onSave(fieldKey, editValue);
    if (success) {
      setIsEditing(false);
    }
  }, [editValue, displayValue, fieldKey, onSave]);

  const handleCancel = useCallback(() => {
    if (blurTimeoutRef.current) {
      clearTimeout(blurTimeoutRef.current);
      blurTimeoutRef.current = null;
    }
    setEditValue(displayValue);
    setIsEditing(false);
  }, [displayValue]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  }, [handleSave, handleCancel]);

  const handleBlur = useCallback(() => {
    // Delay to allow click on confirm button to register first
    blurTimeoutRef.current = setTimeout(() => {
      if (isEditing && !isSaving) {
        handleSave();
      }
    }, 150);
  }, [isEditing, isSaving, handleSave]);

  const handleConfirmClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (blurTimeoutRef.current) {
      clearTimeout(blurTimeoutRef.current);
      blurTimeoutRef.current = null;
    }
    handleSave();
  }, [handleSave]);

  const handleCancelClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    handleCancel();
  }, [handleCancel]);

  const handleDelete = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!onDelete) return;

    setIsDeleting(true);
    try {
      await onDelete(fieldKey);
    } finally {
      setIsDeleting(false);
    }
  }, [fieldKey, onDelete]);

  const handleSelectChange = useCallback(async (newValue: string) => {
    setEditValue(newValue);
    // For select, save immediately on change
    const success = await onSave(fieldKey, newValue);
    if (success) {
      setIsEditing(false);
    }
  }, [fieldKey, onSave]);

  // Render view mode
  if (!isEditing) {
    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="text-muted-foreground text-xs">{label}</label>
          {showDeleteButton && onDelete && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive opacity-40 group-hover:opacity-100 transition-opacity"
              onClick={handleDelete}
              disabled={isDeleting || isSaving}
            >
              {isDeleting ? (
                <Spinner className="h-3 w-3" />
              ) : (
                <Trash2 className="h-3 w-3" />
              )}
            </Button>
          )}
        </div>
        <div
          className={cn(
            "group flex items-center gap-2 py-1.5 px-2 -mx-2 rounded transition-colors min-h-[32px]",
            !readOnly && "cursor-pointer hover:bg-muted/50",
            readOnly && "cursor-default"
          )}
          onClick={handleStartEdit}
        >
          {isSaving ? (
            <Spinner className="h-4 w-4" />
          ) : (
            <>
              <span
                className={cn(
                  "font-medium flex-1 min-w-0 truncate",
                  !displayValue && "text-muted-foreground italic"
                )}
                title={displayValue || undefined}
              >
                {displayValue || 'Not set'}
              </span>
              {!readOnly && (
                <Pencil className="h-3 w-3 text-muted-foreground opacity-40 group-hover:opacity-100 transition-opacity flex-shrink-0" />
              )}
              {readOnly && (
                <Lock className="h-3 w-3 text-muted-foreground opacity-50 flex-shrink-0" />
              )}
            </>
          )}
        </div>
        {error && (
          <p className="text-xs text-destructive animate-in fade-in slide-in-from-top-1">
            {error}
          </p>
        )}
      </div>
    );
  }

  // Render edit mode - Select
  if (type === 'select') {
    return (
      <div className="space-y-1">
        <label className="text-muted-foreground text-xs">{label}</label>
        <div className="flex items-center gap-2">
          <Select
            value={editValue}
            onValueChange={handleSelectChange}
            disabled={isSaving}
          >
            <SelectTrigger className="h-8 flex-1">
              <SelectValue placeholder="Select..." />
            </SelectTrigger>
            <SelectContent>
              {selectOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            size="sm"
            variant="ghost"
            className="h-8 w-8 p-0"
            onClick={handleCancelClick}
            disabled={isSaving}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        {error && (
          <p className="text-xs text-destructive animate-in fade-in slide-in-from-top-1">
            {error}
          </p>
        )}
      </div>
    );
  }

  // Render edit mode - Text input
  return (
    <div className="space-y-1">
      <label className="text-muted-foreground text-xs">{label}</label>
      <div className="flex items-center gap-2">
        <Input
          ref={inputRef}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleBlur}
          className="h-8 flex-1"
          disabled={isSaving}
        />
        <Button
          size="sm"
          variant="ghost"
          className="h-8 w-8 p-0"
          onClick={handleConfirmClick}
          disabled={isSaving}
        >
          {isSaving ? <Spinner className="h-4 w-4" /> : <Check className="h-4 w-4" />}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="h-8 w-8 p-0"
          onClick={handleCancelClick}
          disabled={isSaving}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
      {error && (
        <p className="text-xs text-destructive animate-in fade-in slide-in-from-top-1">
          {error}
        </p>
      )}
    </div>
  );
}
