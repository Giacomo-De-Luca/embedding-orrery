'use client';

import { Tabs, TabsList, TabsTrigger } from '@/lib/ui-primitives/tabs';
import { Database, Upload, Settings, Microscope } from 'lucide-react';

export type DataSourceTab = 'huggingface' | 'local' | 'manage' | 'sae';

interface DataSourceTabsProps {
  activeTab: DataSourceTab;
  onTabChange: (tab: DataSourceTab) => void;
}

export function DataSourceTabs({ activeTab, onTabChange }: DataSourceTabsProps) {
  return (
    <Tabs value={activeTab} onValueChange={(value) => onTabChange(value as DataSourceTab)}>
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="huggingface" className="flex items-center gap-2">
          <Database className="h-4 w-4" />
          <span className="hidden sm:inline">HuggingFace</span>
        </TabsTrigger>
        <TabsTrigger value="local" className="flex items-center gap-2">
          <Upload className="h-4 w-4" />
          <span className="hidden sm:inline">Local Files</span>
        </TabsTrigger>
        <TabsTrigger value="manage" className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          <span className="hidden sm:inline">Manage</span>
        </TabsTrigger>
        <TabsTrigger value="sae" className="flex items-center gap-2">
          <Microscope className="h-4 w-4" />
          <span className="hidden sm:inline">SAE</span>
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
}
