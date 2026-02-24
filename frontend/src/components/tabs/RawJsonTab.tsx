/**
 * Raw JSON Tab
 *
 * Shows the complete pipeline output as formatted JSON.
 * Useful for development and debugging.
 */

import { useState, useEffect } from 'react';
import { Loader2, Copy, Check, Braces } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { getRawJson } from '@/api/client';
import type { RawJsonResponse } from '@/types/api';

interface RawJsonTabProps {
  runId: string;
}

export function RawJsonTab({ runId }: RawJsonTabProps) {
  const [data, setData] = useState<RawJsonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'pipeline' | 'stages'>('stages');
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadRawJson();
  }, [runId]);

  async function loadRawJson() {
    setLoading(true);
    try {
      const response = await getRawJson(runId);
      setData(response);
      // Select first stage by default
      const stageKeys = Object.keys(response.stages_output);
      if (stageKeys.length > 0) {
        setSelectedStage(stageKeys[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load JSON');
    } finally {
      setLoading(false);
    }
  }

  function copyToClipboard(json: object) {
    navigator.clipboard.writeText(JSON.stringify(json, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-full text-destructive">
        {error || 'No data available'}
      </div>
    );
  }

  const stageKeys = Object.keys(data.stages_output);
  const currentJson =
    selectedView === 'pipeline'
      ? data.pipeline_output
      : selectedStage
      ? data.stages_output[selectedStage]
      : {};

  const stageCount = Object.keys(data.stages_output).length;

  return (
    <div className="h-full flex flex-col">
      {/* Contextual Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b bg-muted/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Braces className="h-5 w-5" />
              Raw JSON Output
            </h2>
            <p className="text-sm text-muted-foreground">
              Complete pipeline output data for debugging and development
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{stageCount} Stages</Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Sidebar */}
        <div className="w-48 border-r bg-muted/30">

        <ScrollArea className="h-[calc(100%-4rem)]">
          {/* View Toggle */}
          <div className="p-2 border-b">
            <button
              onClick={() => setSelectedView('stages')}
              className={cn(
                'w-full px-3 py-2 text-left rounded text-sm transition-colors',
                selectedView === 'stages' ? 'bg-accent' : 'hover:bg-accent/50'
              )}
            >
              By Stage
            </button>
            <button
              onClick={() => setSelectedView('pipeline')}
              className={cn(
                'w-full px-3 py-2 text-left rounded text-sm transition-colors',
                selectedView === 'pipeline' ? 'bg-accent' : 'hover:bg-accent/50'
              )}
            >
              Full Pipeline
            </button>
          </div>

          {/* Stage List */}
          {selectedView === 'stages' && (
            <div className="p-2">
              {stageKeys.map((stage) => (
                <button
                  key={stage}
                  onClick={() => setSelectedStage(stage)}
                  className={cn(
                    'w-full px-3 py-2 text-left rounded text-sm transition-colors capitalize',
                    selectedStage === stage ? 'bg-accent' : 'hover:bg-accent/50'
                  )}
                >
                  {stage}
                </button>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>

      {/* JSON Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="p-4 border-b flex items-center justify-between">
          <h3 className="font-semibold capitalize">
            {selectedView === 'pipeline'
              ? 'Full Pipeline Output'
              : selectedStage
              ? `${selectedStage} Stage`
              : 'Select a view'}
          </h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => copyToClipboard(currentJson as object)}
            className="gap-2"
          >
            {copied ? (
              <>
                <Check className="h-4 w-4" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-4 w-4" />
                Copy JSON
              </>
            )}
          </Button>
        </div>

        <ScrollArea className="flex-1">
          <pre className="p-6 text-sm font-mono whitespace-pre-wrap">
            {JSON.stringify(currentJson, null, 2)}
          </pre>
        </ScrollArea>
        </div>
      </div>
    </div>
  );
}
