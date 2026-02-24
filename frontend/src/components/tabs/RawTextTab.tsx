/**
 * Raw Text Tab
 *
 * Shows raw extracted text per page for debugging extraction issues.
 */

import { useState, useEffect } from 'react';
import { Loader2, FileText } from 'lucide-react';

import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { getRawText } from '@/api/client';
import type { RawTextResponse } from '@/types/api';

interface RawTextTabProps {
  runId: string;
}

export function RawTextTab({ runId }: RawTextTabProps) {
  const [data, setData] = useState<RawTextResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState(0);

  useEffect(() => {
    loadRawText();
  }, [runId]);

  async function loadRawText() {
    setLoading(true);
    try {
      const response = await getRawText(runId);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load raw text');
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

  const currentPage = data.pages[selectedPage];

  return (
    <div className="h-full flex flex-col">
      {/* Contextual Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b bg-muted/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Raw Extracted Text
            </h2>
            <p className="text-sm text-muted-foreground">
              {data.total_pages} pages extracted with {data.total_chars.toLocaleString()} total characters
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{data.total_pages} Pages</Badge>
            <Badge variant="outline">{data.total_chars.toLocaleString()} Chars</Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Page List */}
        <div className="w-48 border-r bg-muted/30">

        <ScrollArea className="h-[calc(100%-4rem)]">
          {data.pages.map((page, i) => (
            <button
              key={i}
              onClick={() => setSelectedPage(i)}
              className={cn(
                'w-full px-4 py-2 text-left border-b transition-colors text-sm',
                'hover:bg-accent/50',
                selectedPage === i && 'bg-accent'
              )}
            >
              <div className="flex items-center justify-between">
                <span>Page {page.page_number}</span>
                <span className="text-xs text-muted-foreground">
                  {page.char_count.toLocaleString()}
                </span>
              </div>
            </button>
          ))}
        </ScrollArea>
      </div>

        {/* Page Content */}
        <div className="flex-1 overflow-hidden">
          {currentPage ? (
            <ScrollArea className="h-full">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Page {currentPage.page_number}</h3>
                  <Badge variant="outline">
                    {currentPage.char_count.toLocaleString()} characters
                  </Badge>
                </div>
                <Card>
                  <CardContent className="pt-6">
                    <pre className="text-sm whitespace-pre-wrap font-mono leading-relaxed">
                      {currentPage.text || '(Empty page)'}
                    </pre>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Select a page to view content
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
