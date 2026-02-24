/**
 * Main Application Layout
 *
 * Provides the sidebar + main area structure.
 * The sidebar shows uploaded runs and their status.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Upload, ChevronRight, RefreshCw, AlertCircle, CheckCircle, Clock, Loader2, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { cn, formatDate } from '@/lib/utils';
import { listRuns, uploadPDF, analyzePDF, pollRunStatus, deleteRun } from '@/api/client';
import type { RunListItem } from '@/types/api';

interface LayoutProps {
  children: (props: { onUpload: () => void; uploading: boolean }) => React.ReactNode;
  selectedRunId: string | null;
  onSelectRun: (runId: string | null) => void;
}

export function Layout({ children, selectedRunId, onSelectRun }: LayoutProps) {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load runs on mount and periodically
  useEffect(() => {
    loadRuns();
    const interval = setInterval(loadRuns, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  async function loadRuns() {
    try {
      const response = await listRuns();
      setRuns(response.runs);
      setError(null);
    } catch (err) {
      setError('Failed to load runs');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  // Expose upload trigger for child components
  const triggerUpload = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      // Upload the file
      const uploadResponse = await uploadPDF(file);

      // Start analysis
      const analyzeResponse = await analyzePDF(uploadResponse.file_id);

      // Add to runs list and select it
      const newRun: RunListItem = {
        run_id: analyzeResponse.run_id,
        file_id: analyzeResponse.file_id,
        filename: uploadResponse.filename,
        display_name: null,
        status: 'queued',
        started_at: analyzeResponse.started_at,
        completed_at: null,
        qa_count: null,
        speaker_count: null,
        error_message: null,
      };

      setRuns((prev) => [newRun, ...prev]);
      onSelectRun(analyzeResponse.run_id);

      // Poll for completion
      pollRunStatus(
        analyzeResponse.run_id,
        (summary) => {
          setRuns((prev) =>
            prev.map((r) =>
              r.run_id === summary.run_id
                ? {
                    ...r,
                    status: summary.status,
                    completed_at: summary.completed_at,
                    qa_count: summary.qa_count,
                    speaker_count: summary.speaker_count,
                    error_message: summary.error_message,
                  }
                : r
            )
          );
        },
        2000,
        120
      ).catch(console.error);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      // Reset the input
      event.target.value = '';
    }
  }

  function promptDelete(runId: string, e: React.MouseEvent) {
    e.stopPropagation(); // Don't select the run when clicking delete
    setDeleteTarget(runId);
  }

  async function executeDelete(runId: string) {
    try {
      await deleteRun(runId);
      setRuns((prev) => prev.filter((r) => r.run_id !== runId));

      // Clear selection if the deleted run was selected
      if (selectedRunId === runId) {
        onSelectRun(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete run');
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />;
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin flex-shrink-0" />;
      case 'queued':
        return <Clock className="h-4 w-4 text-yellow-500 flex-shrink-0" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />;
      default:
        return null;
    }
  }

  function getStatusLabel(status: string) {
    switch (status) {
      case 'completed':
        return 'Completed';
      case 'running':
        return 'Analyzing...';
      case 'queued':
        return 'Queued';
      case 'failed':
        return 'Failed';
      default:
        return status;
    }
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={handleUpload}
        disabled={uploading}
      />

      {/* Sidebar */}
      <motion.aside
        initial={{ x: -280 }}
        animate={{ x: 0 }}
        className="w-72 border-r bg-muted/30 flex flex-col flex-shrink-0"
      >
        {/* Header */}
        <div className="p-4 border-b">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="h-6 w-6 text-primary" />
            <h1 className="font-semibold text-lg">Transcript Intel</h1>
          </div>

          {/* Upload Button */}
          <Button
            variant="default"
            className="w-full justify-center gap-2"
            disabled={uploading}
            onClick={triggerUpload}
          >
            {uploading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Upload className="h-4 w-4" />
            )}
            {uploading ? 'Uploading...' : 'Upload PDF'}
          </Button>
        </div>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="px-4 py-2 bg-destructive/10 border-b border-destructive/20"
            >
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                <span className="truncate">{error}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Runs List */}
        <div className="flex-1 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 text-sm text-muted-foreground border-b">
            <span>Recent Runs ({runs.length})</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={loadRuns}
              disabled={loading}
            >
              <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
            </Button>
          </div>

          <ScrollArea className="h-[calc(100%-2.5rem)]">
            <AnimatePresence mode="popLayout">
              {loading && runs.length === 0 ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : runs.length === 0 ? (
                <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                  <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No runs yet</p>
                  <p className="text-xs mt-1">Upload a PDF to get started</p>
                </div>
              ) : (
                runs.map((run) => (
                  <motion.div
                    key={run.run_id}
                    layout
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    className={cn(
                      'w-full px-3 py-3 text-left border-b transition-colors group overflow-hidden',
                      'hover:bg-accent/50 cursor-pointer',
                      selectedRunId === run.run_id && 'bg-accent'
                    )}
                    onClick={() => onSelectRun(run.run_id)}
                  >
                    <div className="flex items-center gap-2">
                      <div className="flex-shrink-0">
                        {getStatusIcon(run.status)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div
                          className="font-medium text-sm truncate"
                          title={run.filename}
                        >
                          {run.filename}
                        </div>
                        <div className="text-xs text-muted-foreground mt-0.5 truncate">
                          {getStatusLabel(run.status)} · {formatDate(run.started_at)}
                        </div>
                        {run.status === 'completed' && (run.speaker_count || run.qa_count) && (
                          <div className="text-xs text-muted-foreground mt-0.5">
                            {run.speaker_count ?? 0} speakers · {run.qa_count ?? 0} Q&As
                          </div>
                        )}
                        {run.error_message && (
                          <div
                            className="text-xs text-destructive mt-0.5 truncate"
                            title={run.error_message}
                          >
                            {run.error_message}
                          </div>
                        )}
                      </div>
                      <div className="flex-shrink-0 flex items-center gap-1">
                        <button
                          onClick={(e) => promptDelete(run.run_id, e)}
                          className="p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
                          title="Delete run"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </ScrollArea>
        </div>

        {/* Footer */}
        <div className="p-4 border-t text-xs text-muted-foreground text-center">
          Transcript Intelligence System v1.0
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        {children({ onUpload: triggerUpload, uploading })}
      </main>

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}
        title="Delete run"
        description="This action cannot be undone. Do you want to proceed?"
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={() => {
          if (deleteTarget) executeDelete(deleteTarget);
        }}
      />
    </div>
  );
}
