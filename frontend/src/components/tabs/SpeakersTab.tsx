/**
 * Speakers Tab
 *
 * Displays the speaker registry with roles, titles, and alias information.
 * Clicking a speaker shows detailed information including LLM reasoning.
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, User, Users, ChevronRight, X, Brain, Merge } from 'lucide-react';

import { SHOW_LLM_DEBUG } from '@/config/features';

import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { cn } from '@/lib/utils';
import { getSpeakers } from '@/api/client';
import type { Speaker, SpeakerRegistryResponse } from '@/types/api';

interface SpeakersTabProps {
  runId: string;
}

export function SpeakersTab({ runId }: SpeakersTabProps) {
  const [data, setData] = useState<SpeakerRegistryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);

  useEffect(() => {
    loadSpeakers();
  }, [runId]);

  async function loadSpeakers() {
    setLoading(true);
    try {
      const response = await getSpeakers(runId);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load speakers');
    } finally {
      setLoading(false);
    }
  }

  function getRoleBadgeVariant(role: string) {
    switch (role) {
      case 'moderator':
        return 'moderator';
      case 'management':
        return 'management';
      case 'analyst':
        return 'analyst';
      default:
        return 'unknown';
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

  return (
    <div className="h-full flex flex-col">
      {/* Contextual Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b bg-muted/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Users className="h-5 w-5" />
              Speaker Registry
            </h2>
            <p className="text-sm text-muted-foreground">
              {data.total_count} identified speakers with roles and affiliations
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="moderator">{data.moderator_count} Moderator</Badge>
            <Badge variant="management">{data.management_count} Management</Badge>
            <Badge variant="analyst">{data.analyst_count} Analyst</Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Main Table */}
        <div className={cn('flex-1 transition-all', selectedSpeaker ? 'pr-96' : '')}>
          <ScrollArea className="h-full">
            <div className="p-6">

            {/* Speakers Table */}
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[250px]">Name</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Title</TableHead>
                    <TableHead>Company</TableHead>
                    <TableHead className="text-right">Turns</TableHead>
                    <TableHead className="w-[50px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.speakers.map((speaker) => (
                    <TableRow
                      key={speaker.speaker_id}
                      className={cn(
                        'cursor-pointer transition-colors',
                        selectedSpeaker?.speaker_id === speaker.speaker_id &&
                          'bg-accent'
                      )}
                      onClick={() => setSelectedSpeaker(speaker)}
                    >
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <User className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium">{speaker.canonical_name}</span>
                          {speaker.aliases.length > 0 && (
                            <Badge variant="outline" className="text-xs">
                              +{speaker.aliases.length} alias
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getRoleBadgeVariant(speaker.role)}>
                          {speaker.role}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {speaker.title || '-'}
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {speaker.company || '-'}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {speaker.turn_count}
                      </TableCell>
                      <TableCell>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </div>
        </ScrollArea>
      </div>

      {/* Detail Panel */}
      <AnimatePresence>
        {selectedSpeaker && (
          <motion.aside
            initial={{ x: 384, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 384, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-96 border-l bg-background shadow-xl z-10"
          >
            <ScrollArea className="h-full">
              <div className="p-6 space-y-6">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">
                      {selectedSpeaker.canonical_name}
                    </h3>
                    <Badge variant={getRoleBadgeVariant(selectedSpeaker.role)} className="mt-1">
                      {selectedSpeaker.role}
                    </Badge>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setSelectedSpeaker(null)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                {/* Details */}
                <div className="space-y-4">
                  {selectedSpeaker.title && (
                    <div>
                      <div className="text-sm text-muted-foreground mb-1">Title</div>
                      <div className="font-medium">{selectedSpeaker.title}</div>
                    </div>
                  )}

                  {selectedSpeaker.company && (
                    <div>
                      <div className="text-sm text-muted-foreground mb-1">Company</div>
                      <div className="font-medium">{selectedSpeaker.company}</div>
                    </div>
                  )}

                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Turn Count</div>
                    <div className="font-medium">{selectedSpeaker.turn_count}</div>
                  </div>

                  {selectedSpeaker.first_appearance_page && (
                    <div>
                      <div className="text-sm text-muted-foreground mb-1">First Appearance</div>
                      <div className="font-medium">Page {selectedSpeaker.first_appearance_page}</div>
                    </div>
                  )}
                </div>

                {/* Aliases */}
                {selectedSpeaker.aliases.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Merge className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Merged Aliases</span>
                    </div>
                    <div className="space-y-2">
                      {selectedSpeaker.aliases.map((alias, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-lg bg-muted/50 text-sm"
                        >
                          <div className="font-medium">{alias.alias}</div>
                          {alias.merge_reason && (
                            <div className="text-muted-foreground text-xs mt-1">
                              {alias.merge_reason}
                            </div>
                          )}
                          {SHOW_LLM_DEBUG && alias.confidence !== null && (
                            <div className="text-muted-foreground text-xs">
                              Confidence: {(alias.confidence * 100).toFixed(0)}%
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* LLM Reasoning (debug only) */}
                {SHOW_LLM_DEBUG && selectedSpeaker.verified_by_llm && (
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Brain className="h-4 w-4 text-amber-500" />
                      <span className="text-sm font-medium">LLM Verification</span>
                    </div>
                    <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 text-sm">
                      {selectedSpeaker.llm_confidence !== null && (
                        <div className="mb-2">
                          <span className="text-muted-foreground">Confidence: </span>
                          <span className="font-medium">
                            {(selectedSpeaker.llm_confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                      {selectedSpeaker.llm_reasoning && (
                        <div className="text-muted-foreground">
                          {selectedSpeaker.llm_reasoning}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </motion.aside>
        )}
      </AnimatePresence>
      </div>
    </div>
  );
}
