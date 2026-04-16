'use client';

import * as React from 'react';
import { Settings2 } from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { Separator } from '@/lib/ui-primitives/separator';
import { VisualizationControls } from './VisualizationControls';
import type { Point2D, Point3D } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { ScrollBar } from '@/lib/ui-primitives/scroll-area';

interface EmbeddingSidebarProps extends React.ComponentProps<typeof Sidebar> {
  embeddingDim: number;
  metadata: {
    pca_2d_variance?: number[];
    pca_3d_variance?: number[];
  };
  selectedPoint: Point2D | Point3D | null;
  colorFieldOptions?: ColorFieldOption[];
  availableFields?: string[];
  nestedColorAvailable?: boolean;
}

export function EmbeddingSidebar({
  embeddingDim,
  metadata,
  selectedPoint,
  colorFieldOptions = [],
  availableFields = [],
  nestedColorAvailable,
  ...props
}: EmbeddingSidebarProps) {
  const { className, ...rest } = props;

  return (
    <Sidebar
      collapsible="offcanvas"
      className={className}
      {...rest}
    >
      <SidebarHeader className="border-b px-4 py-3">
        <div className="flex items-center gap-2">
          {/*<div className="flex size-6 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Settings2 className="size-3.5" />
          </div>*/}
          <span className="font-semibold">Controls</span>
        </div>
      </SidebarHeader>

      <SidebarContent className="gap-0">
        <div className="p-4 space-y-6">
          <VisualizationControls
            embeddingDim={embeddingDim}
            metadata={metadata}
            colorFieldOptions={colorFieldOptions}
            availableFields={availableFields}
            nestedColorAvailable={nestedColorAvailable}
          />

          {selectedPoint && (
            <>
              <Separator />
              {/*<div className="pt-3">
                <SelectedPointCard point={selectedPoint} />
              </div>*/}
            </>
          )}
        </div>
        <ScrollBar orientation="vertical" />

      </SidebarContent>

      <SidebarFooter className="border-t px-4 py-3">
        <div className="text-xs text-muted-foreground text-center">
          Press{' '}
          <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium">
            <span className="text-xs">⌘</span>B
          </kbd>{' '}
          to toggle
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
