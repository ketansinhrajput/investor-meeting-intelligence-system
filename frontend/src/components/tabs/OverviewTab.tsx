/**
 * Overview Tab
 *
 * Shows high-level summary stats and pipeline stage status.
 */

import { motion } from 'framer-motion';
import {
  FileText,
  Users,
  MessageCircle,
  AlertCircle,
  CheckCircle,
  Clock,
  Loader2,
  AlertTriangle,
} from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { formatDate, formatDuration, cn } from '@/lib/utils';
import type { RunSummaryResponse } from '@/types/api';

interface OverviewTabProps {
  summary: RunSummaryResponse;
}

export function OverviewTab({ summary }: OverviewTabProps) {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  function getStageStatusIcon(status: string) {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-muted-foreground" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  }

  return (
    <ScrollArea className="h-full">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="p-6 space-y-6"
      >
        {/* Stats Cards */}
        <motion.div
          variants={itemVariants}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900">
                  <FileText className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <div className="text-2xl font-bold">{summary.page_count}</div>
                  <div className="text-sm text-muted-foreground">Pages</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-purple-100 dark:bg-purple-900">
                  <Users className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <div className="text-2xl font-bold">{summary.speaker_count}</div>
                  <div className="text-sm text-muted-foreground">Speakers</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900">
                  <MessageCircle className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <div className="text-2xl font-bold">{summary.qa_count}</div>
                  <div className="text-sm text-muted-foreground">Q&A Units</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-amber-100 dark:bg-amber-900">
                  <MessageCircle className="h-6 w-6 text-amber-600 dark:text-amber-400" />
                </div>
                <div>
                  <div className="text-2xl font-bold">{summary.follow_up_count}</div>
                  <div className="text-sm text-muted-foreground">Follow-ups</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Status and Pipeline */}
        <motion.div
          variants={itemVariants}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          {/* Run Status */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Run Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Status</span>
                <Badge
                  variant={
                    summary.status === 'completed'
                      ? 'success'
                      : summary.status === 'failed'
                      ? 'error'
                      : summary.status === 'running'
                      ? 'default'
                      : 'warning'
                  }
                >
                  {summary.status}
                </Badge>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Started</span>
                <span className="font-mono text-sm">{formatDate(summary.started_at)}</span>
              </div>

              {summary.completed_at && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Completed</span>
                  <span className="font-mono text-sm">{formatDate(summary.completed_at)}</span>
                </div>
              )}

              {summary.duration_seconds !== null && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Duration</span>
                  <span className="font-mono text-sm">
                    {formatDuration(summary.duration_seconds)}
                  </span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Text Length</span>
                <span className="font-mono text-sm">
                  {summary.total_text_length.toLocaleString()} chars
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Pipeline Stages */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Pipeline Stages</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {summary.stages.map((stage) => (
                  <div
                    key={stage.stage_name}
                    className={cn(
                      'flex items-center justify-between p-3 rounded-lg',
                      stage.status === 'completed' && 'bg-green-50 dark:bg-green-950/30',
                      stage.status === 'running' && 'bg-blue-50 dark:bg-blue-950/30',
                      stage.status === 'failed' && 'bg-red-50 dark:bg-red-950/30',
                      stage.status === 'pending' && 'bg-muted'
                    )}
                  >
                    <div className="flex items-center gap-3">
                      {getStageStatusIcon(stage.status)}
                      <span className="font-medium capitalize">
                        {stage.stage_name.replace('_', ' ')}
                      </span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {stage.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Errors and Warnings */}
        {(summary.errors.length > 0 || summary.warnings.length > 0) && (
          <motion.div variants={itemVariants}>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Issues</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {summary.errors.map((error, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 p-3 rounded-lg bg-red-50 dark:bg-red-950/30"
                  >
                    <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <span className="text-sm text-red-800 dark:text-red-200">{error}</span>
                  </div>
                ))}

                {summary.warnings.map((warning, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 p-3 rounded-lg bg-yellow-50 dark:bg-yellow-950/30"
                  >
                    <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                    <span className="text-sm text-yellow-800 dark:text-yellow-200">{warning}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </motion.div>
    </ScrollArea>
  );
}
