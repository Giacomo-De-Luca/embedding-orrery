'use client';

import { useState, useRef } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Label } from '@/lib/ui-primitives/label';
import { Upload } from 'lucide-react';

interface FileUploadZoneProps {
  filePath: string;
  onFilePathChange: (path: string) => void;
  disabled?: boolean;
}

export function FileUploadZone({ filePath, onFilePathChange, disabled }: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFilePath = (path: string): string | null => {
    if (!path) return null;
    if (!path.startsWith('/')) {
      return 'Must be an absolute path (starting with /)';
    }
    const validExtensions = ['.parquet', '.json', '.jsonl', '.ndjson', '.csv', '.tsv'];
    if (!validExtensions.some(ext => path.toLowerCase().endsWith(ext))) {
      return 'Supported formats: .parquet, .json, .jsonl, .csv, .tsv';
    }
    return null;
  };

  const handleFilePathChange = (path: string) => {
    setValidationError(validateFilePath(path));
    onFilePathChange(path);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled && !isUploading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setValidationError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Determine API URL (assuming localhost:8000 if not configured)
      const apiUrl = 'http://localhost:8000/upload';

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.filePath) {
        handleFilePathChange(data.filePath);
      } else {
        throw new Error('Server returned no file path');
      }
    } catch (err) {
      console.error(err);
      setValidationError(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled || isUploading) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const file = files[0];
      // Check for path (Electron/Local) or Name (Browser)
      const path = (file as any).path;

      if (path && path.startsWith('/')) {
        // It's a real local path (Electron or specific browser config)
        handleFilePathChange(path);
      } else {
        // It's a browser file object, upload it
        await uploadFile(file);
      }
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const path = (file as any).path;
      if (path && path.startsWith('/')) {
        handleFilePathChange(path);
      } else {
        await uploadFile(file);
      }
    }
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-4">
      {/* Drag-drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${isDragging ? 'border-primary bg-primary/5' : 'border-border'}
          ${disabled || isUploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
        onClick={disabled || isUploading ? undefined : handleBrowseClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".parquet,.json,.jsonl,.ndjson,.csv,.tsv"
          onChange={handleFileSelect}
          disabled={disabled || isUploading}
        />
        <div className="flex flex-col items-center gap-2">
          {isUploading ? (
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          ) : (
            <Upload className="h-8 w-8 text-muted-foreground" />
          )}
          <div className="text-sm">
            {isUploading ? (
              <span className="font-medium text-primary">Uploading...</span>
            ) : (
              <>
                <span className="font-medium">Drop file here</span> or{' '}
                <span className="text-primary font-medium">browse</span>
              </>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            Supported: .parquet, .json, .csv, .tsv
          </div>
        </div>
      </div>

      {/* Manual path input */}
      <div className="space-y-2">
        <Label htmlFor="file-path">Or enter absolute file path:</Label>
        <Input
          id="file-path"
          type="text"
          value={filePath}
          onChange={(e) => handleFilePathChange(e.target.value)}
          placeholder="/path/to/your/data.parquet"
          disabled={disabled}
        />
        {validationError && (
          <p className="text-xs text-destructive">{validationError}</p>
        )}
        {filePath && !validationError && (
          <p className="text-xs text-muted-foreground">
            ✓ Valid file path
          </p>
        )}
      </div>

      {/* File type indicators */}
      {filePath && !validationError && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Detected format:</span>
          {filePath.endsWith('.parquet') && <span className="font-medium">Parquet</span>}
          {filePath.endsWith('.json') && <span className="font-medium">JSON</span>}
          {filePath.endsWith('.jsonl') && <span className="font-medium">JSONL</span>}
          {filePath.endsWith('.csv') && <span className="font-medium">CSV</span>}
          {filePath.endsWith('.tsv') && <span className="font-medium">TSV</span>}
        </div>
      )}
    </div>
  );
}
