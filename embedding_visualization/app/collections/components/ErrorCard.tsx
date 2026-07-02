'use client';

import { Button } from '@/lib/ui-primitives/button';
import { Card, CardContent } from '@/lib/ui-primitives/card';

interface ErrorCardProps {
  error: string;
  onDismiss: () => void;
}

export function ErrorCard({ error, onDismiss }: ErrorCardProps) {
  return (
    <Card className="border-destructive">
      <CardContent className="pt-6">
        <div className="text-destructive">
          <strong>Error:</strong> {error}
        </div>
        <Button variant="outline" size="sm" onClick={onDismiss} className="mt-2">
          Dismiss
        </Button>
      </CardContent>
    </Card>
  );
}
