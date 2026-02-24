/**
 * CitationBadge - Clickable citation chip
 *
 * Renders [qa_XXX] or [page_N] references as small clickable badges.
 * Clicking navigates to the relevant tab/item in the parent RunDetail.
 */

import { MessageSquareText, FileText, User } from 'lucide-react';
import type { ChatCitation } from '@/types/api';
import { cn } from '@/lib/utils';

interface CitationBadgeProps {
  citation: ChatCitation;
  onClick?: (citation: ChatCitation) => void;
}

export function CitationBadge({ citation, onClick }: CitationBadgeProps) {
  const iconMap = {
    qa: MessageSquareText,
    page: FileText,
    speaker: User,
  };

  const colorMap = {
    qa: 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/40 dark:text-blue-300',
    page: 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900/40 dark:text-amber-300',
    speaker: 'bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/40 dark:text-green-300',
  };

  const Icon = iconMap[citation.type] || FileText;
  const color = colorMap[citation.type] || colorMap.page;

  return (
    <button
      onClick={() => onClick?.(citation)}
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium',
        'transition-colors cursor-pointer',
        color
      )}
      title={`Go to ${citation.label}`}
    >
      <Icon className="h-3 w-3" />
      {citation.label}
    </button>
  );
}
