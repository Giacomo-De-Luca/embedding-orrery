'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Input } from '@/lib/ui-primitives/input';
import { Button } from '@/lib/ui-primitives/button';
import { Label } from '@/lib/ui-primitives/label';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Plus, X } from 'lucide-react';

// Reserved field names that cannot be used
const RESERVED_FIELDS = new Set([
  'embedding_dim',
  'has_projections',
  'pca_2d_variance',
  'pca_3d_variance',
  'hnsw:space',
  'projections_computed_at',
  'created_at',
]);

export interface AddFieldFormProps {
  /** List of existing field keys to prevent duplicates */
  existingKeys: string[];
  /** Called when a new field is added */
  onAdd: (key: string, value: string) => Promise<boolean>;
  /** Disable the form */
  disabled?: boolean;
}

export function AddFieldForm({
  existingKeys,
  onAdd,
  disabled = false,
}: AddFieldFormProps) {
  const [isAdding, setIsAdding] = useState(false);
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const [keyError, setKeyError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const keyInputRef = useRef<HTMLInputElement>(null);

  // Focus key input when form opens
  useEffect(() => {
    if (isAdding && keyInputRef.current) {
      keyInputRef.current.focus();
    }
  }, [isAdding]);

  const validateKey = useCallback((key: string): string | null => {
    const trimmedKey = key.trim();
    if (!trimmedKey) {
      return 'Field name is required';
    }
    if (existingKeys.includes(trimmedKey)) {
      return 'Field name already exists';
    }
    if (RESERVED_FIELDS.has(trimmedKey)) {
      return 'Cannot use reserved field name';
    }
    // Allow alphanumeric, underscore, hyphen
    if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(trimmedKey)) {
      return 'Field name must start with letter or underscore, and contain only letters, numbers, underscores, or hyphens';
    }
    return null;
  }, [existingKeys]);

  const handleOpen = useCallback(() => {
    setIsAdding(true);
    setNewKey('');
    setNewValue('');
    setKeyError(null);
  }, []);

  const handleCancel = useCallback(() => {
    setIsAdding(false);
    setNewKey('');
    setNewValue('');
    setKeyError(null);
  }, []);

  const handleSubmit = useCallback(async () => {
    const error = validateKey(newKey);
    if (error) {
      setKeyError(error);
      return;
    }

    setIsSaving(true);
    setKeyError(null);

    try {
      const success = await onAdd(newKey.trim(), newValue);
      if (success) {
        setIsAdding(false);
        setNewKey('');
        setNewValue('');
      }
    } finally {
      setIsSaving(false);
    }
  }, [newKey, newValue, validateKey, onAdd]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  }, [handleSubmit, handleCancel]);

  const handleKeyChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setNewKey(e.target.value);
    // Clear error when user starts typing
    if (keyError) {
      setKeyError(null);
    }
  }, [keyError]);

  if (!isAdding) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={handleOpen}
        disabled={disabled}
        className="w-full"
      >
        <Plus className="h-4 w-4 mr-2" />
        Add field
      </Button>
    );
  }

  return (
    <div className="space-y-3 p-3 border rounded-md bg-muted/30">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Add New Field</span>
        <Button
          size="sm"
          variant="ghost"
          className="h-6 w-6 p-0"
          onClick={handleCancel}
          disabled={isSaving}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="space-y-2">
        <div className="space-y-1">
          <Label htmlFor="new-field-key" className="text-xs">
            Field Name
          </Label>
          <Input
            ref={keyInputRef}
            id="new-field-key"
            value={newKey}
            onChange={handleKeyChange}
            onKeyDown={handleKeyDown}
            placeholder="e.g., description"
            className="h-8"
            disabled={isSaving}
          />
          {keyError && (
            <p className="text-xs text-destructive animate-in fade-in slide-in-from-top-1">
              {keyError}
            </p>
          )}
        </div>

        <div className="space-y-1">
          <Label htmlFor="new-field-value" className="text-xs">
            Value
          </Label>
          <Input
            id="new-field-value"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter value..."
            className="h-8"
            disabled={isSaving}
          />
        </div>
      </div>

      <div className="flex justify-end gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={handleCancel}
          disabled={isSaving}
        >
          Cancel
        </Button>
        <Button
          size="sm"
          onClick={handleSubmit}
          disabled={isSaving || !newKey.trim()}
        >
          {isSaving ? <Spinner className="h-4 w-4 mr-2" /> : null}
          Add
        </Button>
      </div>
    </div>
  );
}
