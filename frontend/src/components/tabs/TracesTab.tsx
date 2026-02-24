/**
 * Traces Tab
 *
 * Critical for debugging - shows all LLM decisions with:
 * - Input context
 * - Output decision
 * - Evidence spans
 * - Confidence scores
 */

import { useState, useEffect } from 'react';
import {
  Loader2,
  Brain,
  Zap,
  AlertTriangle,
  Shield,
} from 'lucide-react';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { getTraces } from '@/api/client';
import type { TracesResponse, StageTrace, TraceDecision } from '@/types/api';

interface TracesTabProps {
  runId: string;
}

export function TracesTab({ runId }: TracesTabProps) {
  const [data, setData] = useState<TracesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);

  useEffect(() => {
    loadTraces();
  }, [runId]);

  async function loadTraces() {
    setLoading(true);
    try {
      const response = await getTraces(runId);
      setData(response);
      // Auto-select first stage
      if (response.stages.length > 0) {
        setSelectedStage(response.stages[0].stage_name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load traces');
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

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-full text-destructive">
        {error || 'No data available'}
      </div>
    );
  }

  const selectedStageData = data.stages.find((s) => s.stage_name === selectedStage);

  return (
    <div className="h-full flex flex-col">
      {/* Contextual Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b bg-muted/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Brain className="h-5 w-5 text-amber-500" />
              LLM Decision Traces
            </h2>
            <p className="text-sm text-muted-foreground">
              {data.total_llm_calls} LLM calls across {data.stages.length} pipeline stages
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{data.total_llm_calls} Calls</Badge>
            <Badge variant="outline">{data.stages.length} Stages</Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Stage List Sidebar */}
        <div className="w-64 border-r bg-muted/30">

        <ScrollArea className="h-[calc(100%-4rem)]">
          {data.stages.map((stage) => (
            <button
              key={stage.stage_name}
              onClick={() => setSelectedStage(stage.stage_name)}
              className={cn(
                'w-full px-4 py-3 text-left border-b transition-colors',
                'hover:bg-accent/50',
                selectedStage === stage.stage_name && 'bg-accent'
              )}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium capitalize text-sm">
                  {stage.stage_name}
                </span>
                <Badge variant="outline" className="text-xs">
                  {stage.llm_calls_made} calls
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {stage.decisions.length} decisions
              </div>
            </button>
          ))}
        </ScrollArea>
      </div>

        {/* Stage Details */}
        <div className="flex-1 overflow-hidden">
          {selectedStageData ? (
            <StageTraceView stage={selectedStageData} />
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Select a stage to view traces
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface StageTraceViewProps {
  stage: StageTrace;
}

function StageTraceView({ stage }: StageTraceViewProps) {
  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Stage Header */}
        <div>
          <h3 className="text-lg font-semibold capitalize">{stage.stage_name} Stage</h3>
          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Zap className="h-4 w-4" />
              {stage.llm_calls_made} LLM calls
            </div>
            <div className="flex items-center gap-1">
              <Brain className="h-4 w-4" />
              {stage.decisions.length} decisions
            </div>
            {stage.hard_rules_enforced.length > 0 && (
              <div className="flex items-center gap-1">
                <Shield className="h-4 w-4" />
                {stage.hard_rules_enforced.length} hard rules
              </div>
            )}
          </div>
        </div>

        {/* Warnings */}
        {stage.warnings.length > 0 && (
          <Card className="border-yellow-200 bg-yellow-50/50 dark:border-yellow-800 dark:bg-yellow-950/20">
            <CardHeader className="py-3">
              <CardTitle className="text-sm flex items-center gap-2 text-yellow-700 dark:text-yellow-400">
                <AlertTriangle className="h-4 w-4" />
                Warnings ({stage.warnings.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="py-0 pb-3">
              <ul className="text-sm space-y-1">
                {stage.warnings.map((warning, i) => (
                  <li key={i} className="text-yellow-800 dark:text-yellow-200">
                    • {warning}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Hard Rules */}
        {stage.hard_rules_enforced.length > 0 && (
          <Card className="border-purple-200 bg-purple-50/50 dark:border-purple-800 dark:bg-purple-950/20">
            <CardHeader className="py-3">
              <CardTitle className="text-sm flex items-center gap-2 text-purple-700 dark:text-purple-400">
                <Shield className="h-4 w-4" />
                Hard Rules Enforced ({stage.hard_rules_enforced.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="py-0 pb-3">
              <div className="space-y-2">
                {stage.hard_rules_enforced.map((rule, i) => (
                  <div
                    key={i}
                    className="text-sm p-2 rounded bg-purple-100/50 dark:bg-purple-900/30"
                  >
                    <pre className="text-xs overflow-x-auto">
                      {JSON.stringify(rule, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Decisions */}
        <div>
          <h4 className="font-medium mb-4">LLM Decisions</h4>
          <Accordion type="multiple" className="space-y-2">
            {stage.decisions.map((decision) => (
              <DecisionItem key={decision.decision_id} decision={decision} />
            ))}
          </Accordion>
        </div>
      </div>
    </ScrollArea>
  );
}

interface DecisionItemProps {
  decision: TraceDecision;
}

function DecisionItem({ decision }: DecisionItemProps) {
  return (
    <AccordionItem
      value={decision.decision_id}
      className="border rounded-lg overflow-hidden"
    >
      <AccordionTrigger className="px-4 py-3 hover:no-underline hover:bg-accent/50">
        <div className="flex items-center gap-4 w-full text-left">
          <Badge variant="outline" className="text-xs capitalize">
            {decision.decision_type.replace(/_/g, ' ')}
          </Badge>
          <span className="flex-1 text-sm truncate">{decision.output_decision}</span>
          {decision.confidence !== null && (
            <Badge
              variant={
                decision.confidence >= 0.8
                  ? 'success'
                  : decision.confidence >= 0.5
                  ? 'warning'
                  : 'error'
              }
              className="text-xs"
            >
              {(decision.confidence * 100).toFixed(0)}%
            </Badge>
          )}
        </div>
      </AccordionTrigger>
      <AccordionContent className="px-4 pb-4 space-y-4">
        {/* Input Context */}
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-2">
            INPUT CONTEXT
          </div>
          <div className="p-3 rounded-lg bg-muted text-sm font-mono whitespace-pre-wrap">
            {decision.input_context}
          </div>
        </div>

        {/* Output Decision */}
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-2">
            OUTPUT DECISION
          </div>
          <div className="p-3 rounded-lg bg-green-50 dark:bg-green-950/30 text-sm">
            {decision.output_decision}
          </div>
        </div>

        {/* Reasoning */}
        {decision.reasoning && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">
              REASONING
            </div>
            <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 text-sm">
              {decision.reasoning}
            </div>
          </div>
        )}

        {/* Evidence Spans */}
        {decision.evidence_spans.length > 0 && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">
              EVIDENCE SPANS ({decision.evidence_spans.length})
            </div>
            <div className="space-y-2">
              {decision.evidence_spans.map((span, i) => (
                <div
                  key={i}
                  className="p-3 rounded-lg border bg-card text-sm"
                >
                  <div className="evidence-highlight mb-2">{span.text}</div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span>Source: {span.source}</span>
                    <span>Relevance: {span.relevance}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}
