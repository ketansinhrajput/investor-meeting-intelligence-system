/**
 * Run Detail View
 *
 * Shows the tabbed interface for a selected run.
 * Includes: Overview, Speakers, Q&A, Traces, Raw Text, Raw JSON
 */

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Loader2, AlertCircle, CheckCircle, Clock, RefreshCw, MessageSquare } from 'lucide-react';

import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { OverviewTab } from '@/components/tabs/OverviewTab';
import { SpeakersTab } from '@/components/tabs/SpeakersTab';
import { QAExplorerTab } from '@/components/tabs/QAExplorerTab';
import { AnalystSummaryTab } from '@/components/tabs/AnalystSummaryTab';
import { TracesTab } from '@/components/tabs/TracesTab';
import { RawTextTab } from '@/components/tabs/RawTextTab';
import { RawJsonTab } from '@/components/tabs/RawJsonTab';
import { ChatDrawer } from '@/components/chat/ChatDrawer';
import { getRunSummary } from '@/api/client';
import type { RunSummaryResponse, ChatCitation } from '@/types/api';

interface RunDetailProps {
  runId: string;
}

export function RunDetail({ runId }: RunDetailProps) {
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [chatOpen, setChatOpen] = useState(false);
  const [selectedQaId, setSelectedQaId] = useState<string | null>(null);

  const handleCitationClick = useCallback((citation: ChatCitation) => {
    if (citation.type === 'qa') {
      setSelectedQaId(citation.ref_id);
      setActiveTab('qa');
    } else if (citation.type === 'speaker') {
      setActiveTab('speakers');
    }
    // Close drawer on mobile so user can see the tab
    setChatOpen(false);
  }, []);

  const loadSummary = useCallback(async () => {
    try {
      const data = await getRunSummary(runId);
      setSummary(data);
      setError(null);
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load run');
      return null;
    }
  }, [runId]);

  // Initial load and reset state when runId changes
  useEffect(() => {
    setLoading(true);
    setError(null);
    setSummary(null);
    setActiveTab('overview');

    loadSummary().finally(() => setLoading(false));
  }, [runId, loadSummary]);

  // Auto-refresh when run is in progress
  useEffect(() => {
    if (!summary || summary.status === 'completed' || summary.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      const updated = await loadSummary();
      if (updated?.status === 'completed' || updated?.status === 'failed') {
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [summary?.status, loadSummary]);

  function getStatusBadge(status: string) {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-green-500/10 text-green-600">
            <CheckCircle className="h-3.5 w-3.5" />
            Completed
          </span>
        );
      case 'running':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/10 text-blue-600">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Analyzing...
          </span>
        );
      case 'queued':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-yellow-500/10 text-yellow-600">
            <Clock className="h-3.5 w-3.5" />
            Queued
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-600">
            <AlertCircle className="h-3.5 w-3.5" />
            Failed
          </span>
        );
      default:
        return null;
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading run details...</p>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <div className="text-center">
          <p className="text-destructive font-medium">{error || 'Run not found'}</p>
          <p className="text-sm text-muted-foreground mt-1">
            The run may have been deleted or the server may be unavailable.
          </p>
        </div>
        <Button variant="outline" onClick={() => window.location.reload()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Reload
        </Button>
      </div>
    );
  }

  const isRunning = summary.status === 'running' || summary.status === 'queued';

  return (
    <motion.div
      key={runId}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="h-full flex flex-col"
    >
      {/* Header */}
      <div className="border-b px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-semibold truncate" title={summary.filename}>
                {summary.filename}
              </h2>
              {getStatusBadge(summary.status)}
            </div>
            <div className="text-sm text-muted-foreground mt-1 truncate">
              Run ID: {summary.run_id}
            </div>
            {summary.error_message && (
              <div className="text-sm text-destructive mt-2 p-2 bg-destructive/10 rounded">
                {summary.error_message}
              </div>
            )}
          </div>
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="flex items-center gap-6 text-sm">
              <div className="text-center">
                <div className="text-2xl font-semibold">{summary.page_count ?? '-'}</div>
                <div className="text-xs text-muted-foreground">Pages</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-semibold">{summary.speaker_count ?? '-'}</div>
                <div className="text-xs text-muted-foreground">Speakers</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-semibold">{summary.qa_count ?? '-'}</div>
                <div className="text-xs text-muted-foreground">Q&As</div>
              </div>
            </div>
            {summary.status === 'completed' && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setChatOpen(true)}
                className="gap-1.5"
              >
                <MessageSquare className="h-4 w-4" />
                Ask AI
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* In Progress Message */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-blue-500/10 border-b border-blue-500/20 px-6 py-3"
        >
          <div className="flex items-center gap-2 text-sm text-blue-600">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Analysis in progress. Results will appear as they become available...</span>
          </div>
        </motion.div>
      )}

      {/* Tabs */}
      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="border-b px-6">
          <TabsList className="h-12 bg-transparent p-0 gap-6">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
            >
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="speakers"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
              disabled={isRunning && !summary.speaker_count}
            >
              Speakers
              {summary.speaker_count ? ` (${summary.speaker_count})` : ''}
            </TabsTrigger>
            <TabsTrigger
              value="qa"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
              disabled={isRunning && !summary.qa_count}
            >
              Q&A Explorer
              {summary.qa_count ? ` (${summary.qa_count})` : ''}
            </TabsTrigger>
            <TabsTrigger
              value="analyst-summary"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
              disabled={isRunning && !summary.qa_count}
            >
              Analyst Summary
            </TabsTrigger>
            {/* <TabsTrigger
              value="traces"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
            >
              Traces
            </TabsTrigger>
            <TabsTrigger
              value="raw-text"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
            >
              Raw Text
            </TabsTrigger>
            <TabsTrigger
              value="raw-json"
              className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-3"
            >
              Raw JSON
            </TabsTrigger> */}
          </TabsList>
        </div>

        <div className="flex-1 overflow-hidden">
          <TabsContent value="overview" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <OverviewTab summary={summary} />
          </TabsContent>

          <TabsContent value="speakers" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <SpeakersTab runId={runId} />
          </TabsContent>

          <TabsContent value="qa" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <QAExplorerTab runId={runId} initialSelectedId={selectedQaId} />
          </TabsContent>

          <TabsContent value="analyst-summary" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <AnalystSummaryTab runId={runId} />
          </TabsContent>

          <TabsContent value="traces" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <TracesTab runId={runId} />
          </TabsContent>

          <TabsContent value="raw-text" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <RawTextTab runId={runId} />
          </TabsContent>

          <TabsContent value="raw-json" className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col">
            <RawJsonTab runId={runId} />
          </TabsContent>
        </div>
      </Tabs>

      {/* Chat Drawer */}
      <ChatDrawer
        runId={runId}
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        onCitationClick={handleCitationClick}
      />
    </motion.div>
  );
}
