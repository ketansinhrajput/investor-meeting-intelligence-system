/**
 * ChatMessage - Renders a single chat message bubble
 *
 * User messages: right-aligned, accent color, plain text
 * Assistant messages: left-aligned, markdown-rendered with citation badges
 *
 * Citation approach:
 * 1. Pre-process: replace [qa_XXX] refs with safe placeholders
 * 2. Render through ReactMarkdown
 * 3. Custom components split text children on placeholders and insert CitationBadge
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Clock, Database } from 'lucide-react';
import type { ChatCitation, ChatMessageItem } from '@/types/api';
import { CitationBadge } from './CitationBadge';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  message: ChatMessageItem;
  onCitationClick?: (citation: ChatCitation) => void;
}

// Placeholder format: %%CITE:index%% where index maps to a citation
const CITE_PLACEHOLDER = /%%CITE:(\d+)%%/g;

/**
 * Pre-process content: replace citation patterns with placeholders.
 * Returns { processed content, citation map by index }.
 */
function preprocessCitations(
  content: string,
  citations: ChatCitation[]
): { text: string; citeMap: Map<number, ChatCitation> } {
  if (!citations || citations.length === 0) {
    return { text: content, citeMap: new Map() };
  }

  // Build lookup from ref pattern to citation
  const refToCitation = new Map<string, ChatCitation>();
  for (const c of citations) {
    if (c.type === 'qa') refToCitation.set(`qa_${c.ref_id.replace('qa_', '')}`, c);
    else if (c.type === 'page') refToCitation.set(`page_${c.ref_id}`, c);
    else if (c.type === 'speaker') refToCitation.set(`speaker_${c.ref_id.replace('speaker_', '')}`, c);
  }

  const citeMap = new Map<number, ChatCitation>();
  let citeIndex = 0;

  // Replace [qa_000], [page_5], etc. and fullwidth variants with placeholders
  const pattern = /[\[【](qa_\d+|page_\d+|speaker_\d+)[\]】]/g;
  const text = content.replace(pattern, (match, ref) => {
    const citation = refToCitation.get(ref);
    if (citation) {
      citeMap.set(citeIndex, citation);
      const placeholder = `%%CITE:${citeIndex}%%`;
      citeIndex++;
      return placeholder;
    }
    return match; // Unknown ref, keep as-is
  });

  return { text, citeMap };
}

/**
 * Process React children: find string children containing %%CITE:N%%
 * and split them into text + CitationBadge elements.
 */
function processChildren(
  children: React.ReactNode,
  citeMap: Map<number, ChatCitation>,
  onCitationClick?: (citation: ChatCitation) => void
): React.ReactNode {
  return React.Children.map(children, (child) => {
    if (typeof child !== 'string') return child;

    // Check if this string contains any placeholders
    if (!child.includes('%%CITE:')) return child;

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    const regex = new RegExp(CITE_PLACEHOLDER.source, 'g');

    while ((match = regex.exec(child)) !== null) {
      if (match.index > lastIndex) {
        parts.push(child.slice(lastIndex, match.index));
      }
      const idx = parseInt(match[1], 10);
      const citation = citeMap.get(idx);
      if (citation) {
        parts.push(
          <CitationBadge
            key={`cite-${idx}-${match.index}`}
            citation={citation}
            onClick={onCitationClick}
          />
        );
      }
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < child.length) {
      parts.push(child.slice(lastIndex));
    }
    return <>{parts}</>;
  });
}

/**
 * Build ReactMarkdown custom components that inject CitationBadge
 * into rendered text nodes.
 */
function buildMarkdownComponents(
  citeMap: Map<number, ChatCitation>,
  onCitationClick?: (citation: ChatCitation) => void
) {
  const wrap = (Tag: string) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ({ children, ...props }: any) => {
      return React.createElement(
        Tag,
        props,
        processChildren(children, citeMap, onCitationClick)
      );
    };

  return {
    p: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
      <p className="mb-2 last:mb-0" {...props}>
        {processChildren(children, citeMap, onCitationClick)}
      </p>
    ),
    ul: ({ children, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
      <ul className="ml-4 mb-2 list-disc" {...props}>{children}</ul>
    ),
    ol: ({ children, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
      <ol className="ml-4 mb-2 list-decimal" {...props}>{children}</ol>
    ),
    li: wrap('li'),
    strong: wrap('strong'),
    em: wrap('em'),
    h1: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
      <h3 className="text-sm font-semibold mt-3 mb-1" {...props}>
        {processChildren(children, citeMap, onCitationClick)}
      </h3>
    ),
    h2: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
      <h4 className="text-sm font-semibold mt-2 mb-1" {...props}>
        {processChildren(children, citeMap, onCitationClick)}
      </h4>
    ),
    h3: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
      <h5 className="text-xs font-semibold mt-2 mb-1" {...props}>
        {processChildren(children, citeMap, onCitationClick)}
      </h5>
    ),
    code: ({ children, ...props }: React.HTMLAttributes<HTMLElement>) => (
      <code className="bg-muted px-1 py-0.5 rounded text-xs" {...props}>{children}</code>
    ),
  };
}

function renderAssistantContent(
  content: string,
  citations: ChatCitation[],
  onCitationClick?: (citation: ChatCitation) => void
) {
  const { text, citeMap } = preprocessCitations(content, citations);
  const components = buildMarkdownComponents(citeMap, onCitationClick);

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {text}
    </ReactMarkdown>
  );
}

export function ChatMessage({ message, onCitationClick }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-muted-foreground'
        )}
      >
        {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
      </div>

      {/* Bubble */}
      <div
        className={cn(
          'max-w-[85%] rounded-lg px-3.5 py-2.5 text-sm',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted/60 text-foreground border'
        )}
      >
        {/* Message content */}
        <div className="leading-relaxed">
          {isUser
            ? <span className="whitespace-pre-wrap">{message.content}</span>
            : renderAssistantContent(
                message.content,
                message.citations || [],
                onCitationClick
              )
          }
        </div>

        {/* Assistant metadata footer */}
        {!isUser && (message.total_time_seconds || message.retrieval_source) && (
          <div className="flex items-center gap-3 mt-2 pt-2 border-t border-border/50 text-xs text-muted-foreground">
            {message.total_time_seconds && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {message.total_time_seconds.toFixed(1)}s
              </span>
            )}
            {message.retrieval_source && message.retrieval_source !== 'none' && (
              <span className="flex items-center gap-1">
                <Database className="h-3 w-3" />
                {message.retrieval_source}
              </span>
            )}
          </div>
        )}

        {/* Disclaimer */}
        {!isUser && message.disclaimer && (
          <div className="mt-1.5 text-xs text-muted-foreground italic">
            {message.disclaimer}
          </div>
        )}
      </div>
    </div>
  );
}
