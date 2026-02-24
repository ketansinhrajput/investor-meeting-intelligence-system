/**
 * Analyst Summary Tab
 *
 * Groups questions by questioner name (the only source of truth).
 * No role inference, no AI generation, no backend changes.
 */

import { useState, useEffect, useMemo } from 'react';
import { Loader2, AlertCircle, User, MessageSquare } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { getQAUnits } from '@/api/client';
import type { QAUnit, QAResponse } from '@/types/api';

interface AnalystSummaryTabProps {
  runId: string;
}

interface QuestionerGroup {
  name: string;
  company: string | null;
  questions: { text: string; sequence: number }[];
}

function groupByQuestioner(qaUnits: QAUnit[]): QuestionerGroup[] {
  const map = new Map<string, QuestionerGroup>();

  for (const qa of qaUnits) {
    const key = qa.questioner_name;
    if (!map.has(key)) {
      map.set(key, {
        name: qa.questioner_name,
        company: qa.questioner_company,
        questions: [],
      });
    }
    map.get(key)!.questions.push({
      text: qa.question_text,
      sequence: qa.sequence,
    });
  }

  // Sort: most questions first, then alphabetically
  return Array.from(map.values()).sort((a, b) => {
    if (b.questions.length !== a.questions.length) {
      return b.questions.length - a.questions.length;
    }
    return a.name.localeCompare(b.name);
  });
}

export function AnalystSummaryTab({ runId }: AnalystSummaryTabProps) {
  const [data, setData] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setSelectedName(null);

    getQAUnits(runId)
      .then(setData)
      .catch((err) =>
        setError(err instanceof Error ? err.message : 'Failed to load Q&A data')
      )
      .finally(() => setLoading(false));
  }, [runId]);

  const groups = useMemo(() => {
    if (!data) return [];
    return groupByQuestioner(data.qa_units);
  }, [data]);

  useEffect(() => {
    if (groups.length > 0 && !selectedName) {
      setSelectedName(groups[0].name);
    }
  }, [groups, selectedName]);

  const selectedGroup = useMemo(
    () => groups.find((g) => g.name === selectedName) ?? null,
    [groups, selectedName]
  );

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-destructive">
        <AlertCircle className="h-10 w-10" />
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  // Empty state — no QA data at all
  if (groups.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2">
        <User className="h-10 w-10 opacity-50" />
        <p className="text-sm">No questioners found in this transcript</p>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar — Questioner List */}
      <div className="w-72 border-r flex flex-col flex-shrink-0">
        <div className="px-4 py-3 border-b">
          <h3 className="text-sm font-semibold text-muted-foreground">
            Questioners ({groups.length})
          </h3>
        </div>
        <ScrollArea className="flex-1">
          {groups.map((group) => (
            <button
              key={group.name}
              onClick={() => setSelectedName(group.name)}
              className={cn(
                'w-full text-left px-4 py-3 border-b transition-colors',
                'hover:bg-accent/50',
                selectedName === group.name && 'bg-accent'
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-sm truncate" title={group.name}>
                    {group.name}
                  </div>
                  {group.company && (
                    <div className="text-xs text-muted-foreground truncate mt-0.5">
                      {group.company}
                    </div>
                  )}
                </div>
                <span className="flex-shrink-0 text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                  {group.questions.length}
                </span>
              </div>
            </button>
          ))}
        </ScrollArea>
      </div>

      {/* Main — Question List */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {selectedGroup ? (
          <>
            <div className="px-6 py-4 border-b flex-shrink-0">
              <h2 className="text-lg font-semibold">{selectedGroup.name}</h2>
              <p className="text-sm text-muted-foreground mt-0.5">
                {selectedGroup.questions.length} question
                {selectedGroup.questions.length !== 1 ? 's' : ''}
                {selectedGroup.company ? ` · ${selectedGroup.company}` : ''}
              </p>
            </div>
            <ScrollArea className="flex-1">
              <div className="px-6 py-4 space-y-4">
                {selectedGroup.questions.map((q, idx) => (
                  <div
                    key={q.sequence}
                    className="flex gap-3"
                  >
                    <div className="flex-shrink-0 mt-0.5">
                      <span className="flex items-center justify-center h-6 w-6 rounded-full bg-primary/10 text-primary text-xs font-medium">
                        {idx + 1}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed pt-0.5">{q.text}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
            <MessageSquare className="h-10 w-10 opacity-40" />
            <p className="text-sm">Select a questioner to view their questions</p>
          </div>
        )}
      </div>
    </div>
  );
}
