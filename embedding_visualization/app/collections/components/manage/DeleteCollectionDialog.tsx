'use client';

import { useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/lib/ui-primitives/alert-dialog';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Trash2 } from 'lucide-react';

interface DeleteCollectionDialogProps {
  collectionName: string;
  numItems: number;
  /** Perform the deletion; the dialog closes and shows a spinner on the trigger while pending */
  onConfirm: () => Promise<void>;
  disabled?: boolean;
}

/** Confirmation dialog for the destructive collection delete. */
export function DeleteCollectionDialog({
  collectionName,
  numItems,
  onConfirm,
  disabled,
}: DeleteCollectionDialogProps) {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleConfirm = async () => {
    setIsDeleting(true);
    try {
      await onConfirm();
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="destructive" size="sm" disabled={disabled || isDeleting}>
          {isDeleting ? <Spinner className="h-4 w-4 mr-2" /> : <Trash2 className="h-4 w-4 mr-2" />}
          Delete
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete collection?</AlertDialogTitle>
          <AlertDialogDescription>
            This will remove <strong>{collectionName}</strong> and its{' '}
            {numItems.toLocaleString()} embeddings. This action cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={handleConfirm}>
            Delete collection
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
