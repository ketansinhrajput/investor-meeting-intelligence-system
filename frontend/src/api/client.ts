/**
 * API Client
 *
 * Centralized API calls to the FastAPI backend.
 * All functions return typed responses.
 */

import type {
  UploadResponse,
  AnalyzeResponse,
  RunListResponse,
  RunSummaryResponse,
  SpeakerRegistryResponse,
  QAResponse,
  TracesResponse,
  RawTextResponse,
  RawJsonResponse,
  LoginRequest,
  LoginResponse,
  ChatApiResponse,
  ChatStreamMetadata,
  ChatStreamDone,
} from '@/types/api';

// Base URL for API calls (proxied through Vite in dev)
const API_BASE = '/api';

// =============================================================================
// Token Management
// =============================================================================

const TOKEN_KEY = 'auth_token';

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/**
 * Generic fetch wrapper with error handling and auth injection
 */
async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    clearAuthToken();
    window.dispatchEvent(new CustomEvent('auth:logout'));
    throw new Error('Session expired. Please log in again.');
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = error.detail;
    const message = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `API Error: ${response.status}`;
    throw new Error(message);
  }

  return response.json();
}

// =============================================================================
// Auth
// =============================================================================

export async function login(credentials: LoginRequest): Promise<LoginResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Login failed');
  }

  const data: LoginResponse = await response.json();
  setAuthToken(data.access_token);
  return data;
}

export function logout(): void {
  clearAuthToken();
  window.dispatchEvent(new CustomEvent('auth:logout'));
}

// =============================================================================
// Upload
// =============================================================================

export async function uploadPDF(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const token = getAuthToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (response.status === 401) {
    clearAuthToken();
    window.dispatchEvent(new CustomEvent('auth:logout'));
    throw new Error('Session expired. Please log in again.');
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Upload failed');
  }

  return response.json();
}

// =============================================================================
// Analyze
// =============================================================================

export async function analyzePDF(
  fileId: string,
  skipEnrichment: boolean = false
): Promise<AnalyzeResponse> {
  return apiFetch<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify({ file_id: fileId, skip_enrichment: skipEnrichment }),
  });
}

// =============================================================================
// Runs
// =============================================================================

export async function listRuns(
  limit: number = 50,
  offset: number = 0
): Promise<RunListResponse> {
  return apiFetch<RunListResponse>(`/runs?limit=${limit}&offset=${offset}`);
}

export async function deleteRun(runId: string): Promise<{ status: string; run_id: string }> {
  return apiFetch<{ status: string; run_id: string }>(`/runs/${runId}`, {
    method: 'DELETE',
  });
}

export async function getRunSummary(runId: string): Promise<RunSummaryResponse> {
  return apiFetch<RunSummaryResponse>(`/runs/${runId}/summary`);
}

// =============================================================================
// Speakers
// =============================================================================

export async function getSpeakers(runId: string): Promise<SpeakerRegistryResponse> {
  return apiFetch<SpeakerRegistryResponse>(`/runs/${runId}/speakers`);
}

// =============================================================================
// Q&A
// =============================================================================

export async function getQAUnits(runId: string): Promise<QAResponse> {
  return apiFetch<QAResponse>(`/runs/${runId}/qa`);
}

// =============================================================================
// Traces
// =============================================================================

export async function getTraces(
  runId: string,
  stage?: string
): Promise<TracesResponse> {
  const params = stage ? `?stage=${stage}` : '';
  return apiFetch<TracesResponse>(`/runs/${runId}/traces${params}`);
}

// =============================================================================
// Raw Data
// =============================================================================

export async function getRawText(runId: string): Promise<RawTextResponse> {
  return apiFetch<RawTextResponse>(`/runs/${runId}/raw`);
}

export async function getRawJson(runId: string): Promise<RawJsonResponse> {
  return apiFetch<RawJsonResponse>(`/runs/${runId}/json`);
}

// =============================================================================
// Chat
// =============================================================================

export async function sendChatMessage(
  runId: string,
  message: string,
  history: { role: string; content: string }[] = []
): Promise<ChatApiResponse> {
  return apiFetch<ChatApiResponse>(`/runs/${runId}/chat`, {
    method: 'POST',
    body: JSON.stringify({ message, history }),
  });
}

/**
 * Stream a chat message via SSE (Server-Sent Events).
 *
 * Events:
 *   metadata — {tool_calls, retrieval_source}
 *   token    — {text: "chunk"}
 *   done     — {citations, total_time_seconds, disclaimer, model}
 */
export async function streamChatMessage(
  runId: string,
  message: string,
  history: { role: string; content: string }[],
  callbacks: {
    onMetadata: (data: ChatStreamMetadata) => void;
    onToken: (text: string) => void;
    onDone: (data: ChatStreamDone) => void;
    onError: (error: Error) => void;
  }
): Promise<void> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/runs/${runId}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ message, history }),
    });
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error('Network error'));
    return;
  }

  if (response.status === 401) {
    clearAuthToken();
    window.dispatchEvent(new CustomEvent('auth:logout'));
    callbacks.onError(new Error('Session expired. Please log in again.'));
    return;
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = error.detail;
    const message = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `API Error: ${response.status}`;
    callbacks.onError(new Error(message));
    return;
  }

  if (!response.body) {
    callbacks.onError(new Error('No response body for streaming'));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events (separated by double newlines)
      const events = buffer.split('\n\n');
      // Keep incomplete event in buffer
      buffer = events.pop() || '';

      for (const eventStr of events) {
        if (!eventStr.trim()) continue;

        const lines = eventStr.trim().split('\n');
        let eventType = '';
        let data = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            data = line.slice(6);
          }
        }

        if (!eventType || !data) continue;

        try {
          const parsed = JSON.parse(data);

          switch (eventType) {
            case 'metadata':
              callbacks.onMetadata(parsed as ChatStreamMetadata);
              break;
            case 'token':
              callbacks.onToken(parsed.text || '');
              break;
            case 'done':
              callbacks.onDone(parsed as ChatStreamDone);
              break;
            case 'error':
              callbacks.onError(new Error(typeof parsed.error === 'string' ? parsed.error : JSON.stringify(parsed.error) || 'Stream error'));
              break;
          }
        } catch {
          // Malformed JSON in SSE data — skip
        }
      }
    }
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error('Stream reading failed'));
  }
}

// =============================================================================
// Polling Helper
// =============================================================================

/**
 * Poll a run until it completes or fails
 */
export async function pollRunStatus(
  runId: string,
  onUpdate: (summary: RunSummaryResponse) => void,
  intervalMs: number = 2000,
  maxAttempts: number = 60
): Promise<RunSummaryResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    const summary = await getRunSummary(runId);
    onUpdate(summary);

    if (summary.status === 'completed' || summary.status === 'failed') {
      return summary;
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error('Polling timeout');
}
