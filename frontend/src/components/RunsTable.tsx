/**
 * Runs Table Component
 *
 * Displays all analysis runs in a clean table format.
 * Default sort: most recently uploaded first.
 */

import { useState, useEffect, useMemo } from 'react';
import {
  Loader2,
  RefreshCw,
  Trash2,
  CheckCircle,
  Clock,
  AlertCircle,
  Eye,
  ArrowUp,
  ArrowDown,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { listRuns, deleteRun } from '@/api/client';
import type { RunListItem } from '@/types/api';
import { formatDate } from '@/lib/utils';

interface RunsTableProps {
  onSelectRun: (runId: string) => void;
  refreshTrigger?: number;
}

type SortDir = 'asc' | 'desc';

export function RunsTable({ onSelectRun, refreshTrigger }: RunsTableProps) {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  useEffect(() => {
    loadRuns();
  }, [refreshTrigger]);

  // Auto-refresh for running jobs
  useEffect(() => {
    const hasRunning = runs.some(r => r.status === 'running' || r.status === 'queued');
    if (!hasRunning) return;

    const interval = setInterval(loadRuns, 3000);
    return () => clearInterval(interval);
  }, [runs]);

  const sortedRuns = useMemo(() => {
    return [...runs].sort((a, b) => {
      const dateA = new Date(a.started_at).getTime();
      const dateB = new Date(b.started_at).getTime();
      return sortDir === 'desc' ? dateB - dateA : dateA - dateB;
    });
  }, [runs, sortDir]);

  async function loadRuns() {
    try {
      const response = await listRuns();
      setRuns(response.runs);
    } catch (err) {
      console.error('Failed to load runs:', err);
    } finally {
      setLoading(false);
    }
  }

  async function executeDelete(runId: string) {
    setDeleting(runId);
    try {
      await deleteRun(runId);
      setRuns(prev => prev.filter(r => r.run_id !== runId));
    } catch (err) {
      console.error('Failed to delete:', err);
    } finally {
      setDeleting(null);
    }
  }

  function toggleSort() {
    setSortDir(prev => (prev === 'desc' ? 'asc' : 'desc'));
  }

  function getStatusBadge(status: string) {
    switch (status) {
      case 'completed':
        return (
          <Badge variant="default" className="bg-green-500 hover:bg-green-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        );
      case 'running':
        return (
          <Badge variant="default" className="bg-blue-500 hover:bg-blue-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Running
          </Badge>
        );
      case 'queued':
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            Queued
          </Badge>
        );
      case 'failed':
        return (
          <Badge variant="destructive">
            <AlertCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <p className="text-lg mb-2">No analysis runs yet</p>
        <p className="text-sm">Upload a PDF to get started</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Analysis Runs ({runs.length})</h2>
        <Button variant="outline" size="sm" onClick={loadRuns}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[300px]">Filename</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>
                <button
                  onClick={toggleSort}
                  className="flex items-center gap-1 hover:text-foreground transition-colors"
                >
                  Started
                  {sortDir === 'desc' ? (
                    <ArrowDown className="h-3 w-3" />
                  ) : (
                    <ArrowUp className="h-3 w-3" />
                  )}
                </button>
              </TableHead>
              <TableHead className="text-center">Speakers</TableHead>
              <TableHead className="text-center">Q&As</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedRuns.map((run) => (
              <TableRow key={run.run_id}>
                <TableCell className="font-medium">
                  <div className="max-w-[280px] truncate" title={run.filename}>
                    {run.filename}
                  </div>
                  {run.error_message && (
                    <div className="text-xs text-destructive truncate mt-1" title={run.error_message}>
                      {run.error_message}
                    </div>
                  )}
                </TableCell>
                <TableCell>{getStatusBadge(run.status)}</TableCell>
                <TableCell className="text-muted-foreground text-sm">
                  {formatDate(run.started_at)}
                </TableCell>
                <TableCell className="text-center">
                  {run.speaker_count ?? '-'}
                </TableCell>
                <TableCell className="text-center">
                  {run.qa_count ?? '-'}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onSelectRun(run.run_id)}
                      disabled={run.status === 'queued'}
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      View
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeleteTarget(run.run_id)}
                      disabled={deleting === run.run_id}
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    >
                      {deleting === run.run_id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

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
