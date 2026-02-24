/**
 * Q&A Explorer Tab
 *
 * Split layout view for Q&A units with:
 * - Left sidebar: Scrollable list of Q&A items
 * - Right panel: Detailed view of selected Q&A
 * - Follow-up chains visualization
 * - LLM boundary reasoning and enrichment data
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Loader2,
  MessageCircle,
  Link,
  User,
  FileText,
  Brain,
  Tag,
  Target,
  Shield,
  Building,
  MessageSquare,
  ArrowRight,
} from 'lucide-react';

import { SHOW_LLM_DEBUG } from '@/config/features';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn, truncate } from '@/lib/utils';
import { getQAUnits } from '@/api/client';
import type { QAUnit, QAResponse } from '@/types/api';

interface QAExplorerTabProps {
  runId: string;
  initialSelectedId?: string | null;
}

export function QAExplorerTab({ runId, initialSelectedId }: QAExplorerTabProps) {
  const [data, setData] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    loadQAUnits();
  }, [runId]);

  useEffect(() => {
    // Auto-select first Q&A when data loads
    if (data?.qa_units.length && !selectedId) {
      setSelectedId(data.qa_units[0].qa_id);
    }
  }, [data, selectedId]);

  // Handle external citation navigation (e.g., from chat drawer)
  useEffect(() => {
    if (initialSelectedId && data?.qa_units.some(q => q.qa_id === initialSelectedId)) {
      setSelectedId(initialSelectedId);
      // Scroll the sidebar item into view after a brief delay for render
      setTimeout(() => {
        document.querySelector(`[data-qa-id="${initialSelectedId}"]`)
          ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    }
  }, [initialSelectedId, data]);

  async function loadQAUnits() {
    setLoading(true);
    setSelectedId(null);
    try {
      const response = await getQAUnits(runId);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load Q&A units');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <Card className="max-w-md">
          <CardContent className="pt-6">
            <div className="text-center text-destructive">
              <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="font-medium">Failed to load Q&A data</p>
              <p className="text-sm text-muted-foreground mt-1">{error}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data || data.qa_units.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <Card className="max-w-md">
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-30" />
              <p className="font-medium">No Q&A units found</p>
              <p className="text-sm mt-1">
                This transcript may not contain a Q&A session, or the extraction is still in progress.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const selectedQA = data.qa_units.find((qa) => qa.qa_id === selectedId);
  const followUps = selectedQA
    ? selectedQA.follow_up_ids
        .map((id) => data.qa_units.find((q) => q.qa_id === id))
        .filter((q): q is QAUnit => q !== undefined)
    : [];

  return (
    <TooltipProvider>
      <div className="h-full flex flex-col">
        {/* Contextual Header */}
        <div className="flex-shrink-0 px-6 py-4 border-b bg-muted/30">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                Q&A Explorer
              </h2>
              <p className="text-sm text-muted-foreground">
                {data.total_count} questions from {data.unique_questioners} analysts
                {data.follow_up_count > 0 && ` • ${data.follow_up_count} follow-ups`}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">{data.total_count} Q&As</Badge>
              <Badge variant="outline">{data.follow_up_count} Follow-ups</Badge>
            </div>
          </div>
        </div>

        {/* Split Layout */}
        <div className="flex-1 flex min-h-0">
          {/* Left Sidebar - Q&A List */}
          <div className="w-80 flex-shrink-0 border-r">
            <ScrollArea className="h-full">
              <div className="p-2">
                {data.qa_units.map((qa) => (
                  <QAListItem
                    key={qa.qa_id}
                    qa={qa}
                    isSelected={selectedId === qa.qa_id}
                    onSelect={() => setSelectedId(qa.qa_id)}
                  />
                ))}
              </div>
            </ScrollArea>
          </div>

          {/* Right Panel - Detail View */}
          <div className="flex-1 min-w-0">
            <ScrollArea className="h-full">
              <AnimatePresence mode="wait">
                {selectedQA ? (
                  <motion.div
                    key={selectedQA.qa_id}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.2 }}
                    className="p-6"
                  >
                    <QADetailPanel qa={selectedQA} followUps={followUps} />
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-center justify-center h-full text-muted-foreground"
                  >
                    <div className="text-center">
                      <ArrowRight className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      <p>Select a Q&A from the list</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </ScrollArea>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// ============================================================================
// QA List Item (Sidebar)
// ============================================================================

interface QAListItemProps {
  qa: QAUnit;
  isSelected: boolean;
  onSelect: () => void;
}

function QAListItem({ qa, isSelected, onSelect }: QAListItemProps) {
  return (
    <motion.button
      onClick={onSelect}
      data-qa-id={qa.qa_id}
      className={cn(
        'w-full text-left p-3 rounded-lg mb-1 transition-all',
        'hover:bg-accent/50',
        isSelected && 'bg-accent shadow-sm',
        qa.is_follow_up && 'ml-4 border-l-2 border-l-amber-400'
      )}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className="flex items-start gap-3">
        {/* Question Number */}
        <div
          className={cn(
            'flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium',
            isSelected
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground'
          )}
        >
          {qa.sequence + 1}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Questioner Name */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium truncate">
              {qa.questioner_name}
            </span>
            {qa.is_follow_up && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="warning" className="text-[10px] px-1.5 py-0">
                    <Link className="h-2.5 w-2.5" />
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>Follow-up question</TooltipContent>
              </Tooltip>
            )}
            {qa.has_follow_ups && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                    +{qa.follow_up_ids.length}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  {qa.follow_up_ids.length} follow-up question{qa.follow_up_ids.length > 1 ? 's' : ''}
                </TooltipContent>
              </Tooltip>
            )}
          </div>

          {/* Question Preview */}
          <p className="text-xs text-muted-foreground line-clamp-2">
            {truncate(qa.question_text, 100)}
          </p>

          {/* Topics Preview */}
          {qa.topics && qa.topics.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {qa.topics.slice(0, 2).map((topic, i) => (
                <Badge key={i} variant="outline" className="text-[10px] px-1.5 py-0">
                  {topic}
                </Badge>
              ))}
              {qa.topics.length > 2 && (
                <span className="text-[10px] text-muted-foreground">
                  +{qa.topics.length - 2}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.button>
  );
}

// ============================================================================
// QA Detail Panel (Main Content)
// ============================================================================

interface QADetailPanelProps {
  qa: QAUnit;
  followUps: QAUnit[];
}

function QADetailPanel({ qa, followUps }: QADetailPanelProps) {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Badge variant="outline">#{qa.sequence + 1}</Badge>
            {qa.is_follow_up && (
              <Badge variant="warning">
                <Link className="h-3 w-3 mr-1" />
                Follow-up
              </Badge>
            )}
            {(qa.start_page || qa.end_page) && (
              <span className="text-xs text-muted-foreground flex items-center gap-1">
                <FileText className="h-3 w-3" />
                Page {qa.start_page}
                {qa.end_page !== qa.start_page && `-${qa.end_page}`}
              </span>
            )}
          </div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <User className="h-5 w-5 text-green-600" />
            {qa.questioner_name}
            {qa.questioner_company && (
              <span className="text-muted-foreground font-normal text-base flex items-center gap-1">
                <Building className="h-4 w-4" />
                {qa.questioner_company}
              </span>
            )}
          </h3>
        </div>
      </div>

      {/* Question Section */}
      <Card className="border-l-4 border-l-green-500">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-green-700 dark:text-green-400 flex items-center gap-2">
            <MessageCircle className="h-4 w-4" />
            Question
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm leading-relaxed whitespace-pre-wrap">{qa.question_text}</div>
        </CardContent>
      </Card>

      {/* Response Section */}
      {qa.response_text && (
        <Card className="border-l-4 border-l-blue-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-blue-700 dark:text-blue-400 flex items-center gap-2">
              <MessageCircle className="h-4 w-4" />
              Response from {qa.responder_names.join(', ') || 'Management'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm leading-relaxed whitespace-pre-wrap">{qa.response_text}</div>
          </CardContent>
        </Card>
      )}

      {/* Enrichment Data */}
      {(qa.topics?.length > 0 || qa.investor_intent || qa.response_posture) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Tag className="h-4 w-4 text-purple-500" />
              Insights
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Topics */}
              {qa.topics && qa.topics.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 text-sm font-medium mb-2">
                    <Tag className="h-4 w-4 text-purple-500" />
                    Topics
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {qa.topics.map((topic, i) => (
                      <Badge key={i} variant="secondary" className="text-xs">
                        {topic}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Investor Intent */}
              {qa.investor_intent && (
                <div>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2 text-sm font-medium mb-2 cursor-help">
                        <Target className="h-4 w-4 text-blue-500" />
                        Question Intent
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      The underlying purpose behind the analyst's question
                    </TooltipContent>
                  </Tooltip>
                  <Badge variant="outline" className="capitalize">
                    {qa.investor_intent}
                  </Badge>
                </div>
              )}

              {/* Response Posture */}
              {qa.response_posture && (
                <div>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2 text-sm font-medium mb-2 cursor-help">
                        <Shield className="h-4 w-4 text-green-500" />
                        Response Tone
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      How management approached and framed their response
                    </TooltipContent>
                  </Tooltip>
                  <Badge variant="outline" className="capitalize">
                    {qa.response_posture}
                  </Badge>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* LLM Boundary Reasoning (debug only) */}
      {SHOW_LLM_DEBUG && qa.boundary_reasoning && (
        <Card className="border-l-4 border-l-amber-400">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-amber-700 dark:text-amber-400 flex items-center gap-2">
              <Brain className="h-4 w-4" />
              LLM Boundary Decision
            </CardTitle>
            <CardDescription>
              How the AI determined where this Q&A unit begins and ends
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground">{qa.boundary_reasoning}</div>
          </CardContent>
        </Card>
      )}

      {/* Follow-ups */}
      {followUps.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Link className="h-4 w-4 text-amber-500" />
              Follow-up Questions ({followUps.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {followUps.map((fu) => (
                <div
                  key={fu.qa_id}
                  className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 text-sm"
                >
                  <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400 mb-1">
                    <User className="h-3 w-3" />
                    <span className="font-medium">{fu.questioner_name}</span>
                  </div>
                  <div className="text-muted-foreground">{truncate(fu.question_text, 200)}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
