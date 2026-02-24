/**
 * ChatDrawer - Slide-in chat panel for conversational Q&A
 *
 * Opens from the right side of RunDetail. Maintains per-run chat history
 * in sessionStorage (survives page refresh, clears on tab close).
 * Uses SSE streaming for progressive token display.
 *
 * Citation clicks emit a callback so RunDetail can navigate to the
 * relevant tab (Q&A Explorer, Speakers, etc.).
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, MessageSquare, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { streamChatMessage, sendChatMessage } from '@/api/client';
import type { ChatCitation, ChatMessageItem } from '@/types/api';

const STORAGE_PREFIX = 'chat_history_';
const MAX_STORED_MESSAGES = 50;

function getStorageKey(runId: string) {
  return `${STORAGE_PREFIX}${runId}`;
}

function loadStoredMessages(runId: string): ChatMessageItem[] {
  try {
    const stored = sessionStorage.getItem(getStorageKey(runId));
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveMessages(runId: string, messages: ChatMessageItem[]) {
  try {
    if (messages.length > 0) {
      const toStore = messages.slice(-MAX_STORED_MESSAGES);
      sessionStorage.setItem(getStorageKey(runId), JSON.stringify(toStore));
    } else {
      sessionStorage.removeItem(getStorageKey(runId));
    }
  } catch {
    // sessionStorage full or unavailable
  }
}

interface ChatDrawerProps {
  runId: string;
  isOpen: boolean;
  onClose: () => void;
  onCitationClick?: (citation: ChatCitation) => void;
}

export function ChatDrawer({ runId, isOpen, onClose, onCitationClick }: ChatDrawerProps) {
  const [messages, setMessages] = useState<ChatMessageItem[]>(() => loadStoredMessages(runId));
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const prevRunId = useRef(runId);
  // Accumulate streaming tokens in a ref to batch state updates
  const streamBufferRef = useRef('');
  const rafRef = useRef<number | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  // Load history when run changes
  useEffect(() => {
    if (prevRunId.current !== runId) {
      setMessages(loadStoredMessages(runId));
      prevRunId.current = runId;
    }
  }, [runId]);

  // Persist messages to sessionStorage on change
  useEffect(() => {
    saveMessages(runId, messages);
  }, [messages, runId]);

  // Smooth auto-scroll to bottom — target the Radix viewport directly
  const scrollToBottom = useCallback(() => {
    // Radix ScrollArea puts a [data-radix-scroll-area-viewport] inside
    const viewport = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]');
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight;
    } else {
      // Fallback to scrollIntoView
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, isOpen, scrollToBottom]);

  // Flush streaming buffer to state (batched via requestAnimationFrame)
  const flushStreamBuffer = useCallback(() => {
    const text = streamBufferRef.current;
    if (!text) return;
    streamBufferRef.current = '';

    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last && last.role === 'assistant') {
        updated[updated.length - 1] = { ...last, content: last.content + text };
      }
      return updated;
    });
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      const userMessage: ChatMessageItem = { role: 'user', content: text };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      // Build history from previous messages (max 20 to match backend validation)
      const history = messages.slice(-20).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Create empty assistant placeholder for streaming
      const placeholder: ChatMessageItem = {
        role: 'assistant',
        content: '',
      };
      setMessages((prev) => [...prev, placeholder]);

      try {
        await streamChatMessage(runId, text, history, {
          onMetadata: (data) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  tool_calls: data.tool_calls,
                  retrieval_source: data.retrieval_source,
                };
              }
              return updated;
            });
          },
          onToken: (tokenText) => {
            // Batch token updates via requestAnimationFrame for performance
            streamBufferRef.current += tokenText;
            if (rafRef.current === null) {
              rafRef.current = requestAnimationFrame(() => {
                rafRef.current = null;
                flushStreamBuffer();
              });
            }
          },
          onDone: (data) => {
            // Flush any remaining buffered tokens
            flushStreamBuffer();

            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  citations: data.citations,
                  total_time_seconds: data.total_time_seconds,
                  disclaimer: data.disclaimer,
                };
              }
              return updated;
            });
            setIsLoading(false);
          },
          onError: (error) => {
            // Flush any buffered tokens first
            flushStreamBuffer();

            const errorMsg = error instanceof Error ? error.message : String(error || 'Unknown error');
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === 'assistant' && !last.content) {
                // Empty placeholder - replace with error
                updated[updated.length - 1] = {
                  ...last,
                  content: `Error: ${errorMsg}`,
                };
              } else {
                // Had partial content - append error
                updated.push({
                  role: 'assistant',
                  content: `Error: ${errorMsg}`,
                });
              }
              return updated;
            });
            setIsLoading(false);
          },
        });
      } catch {
        // Fallback: if streaming fails entirely, try non-streaming
        try {
          const response = await sendChatMessage(runId, text, history);
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: 'assistant',
              content: response.answer,
              citations: response.citations,
              tool_calls: response.tool_calls,
              retrieval_source: response.retrieval_source,
              total_time_seconds: response.total_time_seconds,
              disclaimer: response.disclaimer,
            };
            return updated;
          });
        } catch (err) {
          const fallbackMsg = err instanceof Error ? err.message : typeof err === 'string' ? err : 'An unexpected error occurred.';
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: 'assistant',
              content: `Error: ${fallbackMsg}`,
            };
            return updated;
          });
        } finally {
          setIsLoading(false);
        }
      }
    },
    [runId, messages, flushStreamBuffer]
  );

  const handleClear = useCallback(() => {
    setMessages([]);
    sessionStorage.removeItem(getStorageKey(runId));
  }, [runId]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 z-40"
            onClick={onClose}
          />

          {/* Drawer panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed right-0 top-0 bottom-0 w-full bg-background border-l shadow-xl z-50 flex flex-col"
            style={{ maxWidth: '50rem' }}
            onKeyDown={(e) => {
              // Auto-focus textarea on printable key press anywhere in drawer
              if (e.key.length === 1 && document.activeElement !== chatInputRef.current) {
                chatInputRef.current?.focus();
              }
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4 text-primary" />
                <h3 className="font-semibold text-sm">Ask AI</h3>
                {messages.length > 0 && (
                  <span className="text-xs text-muted-foreground">
                    ({messages.length} messages)
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1">
                {messages.length > 0 && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={handleClear}
                    title="Clear chat"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={onClose}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Messages area */}
            <ScrollArea className="flex-1" ref={scrollAreaRef}>
              <div className="p-4 space-y-4">
                {messages.length === 0 && (
                  <div className="text-center text-muted-foreground text-sm py-12">
                    <MessageSquare className="h-8 w-8 mx-auto mb-3 opacity-40" />
                    <p className="font-medium">Ask questions about this transcript</p>
                    <p className="text-xs mt-1.5 max-w-[250px] mx-auto">
                      Try: "What were the key topics discussed?" or "Summarize the Q&A session"
                    </p>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <ChatMessage
                    key={i}
                    message={msg}
                    onCitationClick={onCitationClick}
                  />
                ))}

                {/* Loading indicator (shown before first token arrives) */}
                {isLoading && messages[messages.length - 1]?.content === '' && (
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-muted text-muted-foreground">
                      <MessageSquare className="h-3.5 w-3.5" />
                    </div>
                    <div className="bg-muted/60 border rounded-lg px-3.5 py-2.5">
                      <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:-0.3s]" />
                        <div className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:-0.15s]" />
                        <div className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full animate-bounce" />
                      </div>
                    </div>
                  </div>
                )}

                {/* Scroll sentinel */}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>

            {/* Input */}
            <ChatInput onSend={handleSend} isLoading={isLoading} inputRef={chatInputRef} />
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
