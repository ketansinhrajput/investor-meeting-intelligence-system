/**
 * ChatInput - Auto-expanding textarea with send button
 *
 * Features:
 * - Auto-expands up to 4 lines
 * - Enter to send, Shift+Enter for newline
 * - Disabled while loading
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  placeholder?: string;
  inputRef?: React.RefObject<HTMLTextAreaElement | null>;
}

export function ChatInput({ onSend, isLoading, placeholder = 'Ask about this transcript...', inputRef }: ChatInputProps) {
  const [value, setValue] = useState('');
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const textareaRef = inputRef || internalRef;

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`; // max ~4 lines
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  // Auto-focus on mount (drawer open)
  useEffect(() => {
    const timer = setTimeout(() => {
      textareaRef.current?.focus();
    }, 100); // slight delay for drawer animation
    return () => clearTimeout(timer);
  }, []);

  // Re-focus when loading completes (answer received)
  const prevLoading = useRef(isLoading);
  useEffect(() => {
    if (prevLoading.current && !isLoading) {
      textareaRef.current?.focus();
    }
    prevLoading.current = isLoading;
  }, [isLoading]);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setValue('');
    // Reset height and re-focus after clearing
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.focus();
      }
    }, 0);
  }, [value, isLoading, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="flex items-end gap-2 p-3 border-t bg-background">
      <textarea
        ref={textareaRef as React.RefObject<HTMLTextAreaElement>}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isLoading}
        rows={1}
        className={cn(
          'flex-1 resize-none rounded-lg border bg-background px-3 py-2 text-sm',
          'placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'min-h-[38px] max-h-[120px]'
        )}
      />
      <Button
        size="icon"
        onClick={handleSend}
        disabled={!value.trim() || isLoading}
        className="h-[38px] w-[38px] flex-shrink-0"
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Send className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}
