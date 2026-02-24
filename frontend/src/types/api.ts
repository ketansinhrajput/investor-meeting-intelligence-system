/**
 * TypeScript types for API responses.
 *
 * These mirror the Pydantic schemas from the backend.
 * Keep in sync with backend/api/schemas/responses.py
 */

// =============================================================================
// Auth
// =============================================================================

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  username: string;
  role: string;
}

export interface AuthUser {
  username: string;
  role: string;
}

// =============================================================================
// Common Types
// =============================================================================

export type RunStatus = 'queued' | 'running' | 'completed' | 'failed';
export type SpeakerRole = 'moderator' | 'management' | 'analyst' | 'unknown';
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

// =============================================================================
// Upload
// =============================================================================

export interface UploadResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
  page_count: number | null;
  upload_time: string;
}

// =============================================================================
// Analyze
// =============================================================================

export interface AnalyzeResponse {
  run_id: string;
  file_id: string;
  status: RunStatus;
  started_at: string;
}

// =============================================================================
// Run List
// =============================================================================

export interface RunListItem {
  run_id: string;
  file_id: string;
  filename: string;
  display_name: string | null;
  status: RunStatus;
  started_at: string;
  completed_at: string | null;
  qa_count: number | null;
  speaker_count: number | null;
  error_message: string | null;
}

export interface RunListResponse {
  runs: RunListItem[];
  total_count: number;
}

// =============================================================================
// Run Summary
// =============================================================================

export interface PipelineStageStatus {
  stage_name: string;
  status: StageStatus;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  warnings: string[];
}

export interface RunSummaryResponse {
  run_id: string;
  file_id: string;
  filename: string;
  display_name: string | null;
  status: RunStatus;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  page_count: number;
  total_text_length: number;
  speaker_count: number;
  qa_count: number;
  follow_up_count: number;
  strategic_statement_count: number;
  stages: PipelineStageStatus[];
  errors: string[];
  warnings: string[];
  error_message: string | null;
}

// =============================================================================
// Speakers
// =============================================================================

export interface SpeakerAlias {
  alias: string;
  merge_reason: string | null;
  confidence: number | null;
}

export interface Speaker {
  speaker_id: string;
  canonical_name: string;
  role: SpeakerRole;
  title: string | null;
  company: string | null;
  turn_count: number;
  first_appearance_page: number | null;
  aliases: SpeakerAlias[];
  verified_by_llm: boolean;
  llm_confidence: number | null;
  llm_reasoning: string | null;
}

export interface SpeakerRegistryResponse {
  run_id: string;
  speakers: Speaker[];
  total_count: number;
  management_count: number;
  analyst_count: number;
  moderator_count: number;
}

// =============================================================================
// Q&A Units
// =============================================================================

export interface SpeakerTurn {
  speaker_name: string;
  speaker_id: string | null;
  text: string;
  page_number: number | null;
  is_question: boolean;
}

export interface QAUnit {
  qa_id: string;
  sequence: number;
  questioner_name: string;
  questioner_id: string | null;
  questioner_company: string | null;
  question_text: string;
  question_turns: SpeakerTurn[];
  responder_names: string[];
  responder_ids: string[];
  response_text: string;
  response_turns: SpeakerTurn[];
  is_follow_up: boolean;
  follow_up_of: string | null;
  has_follow_ups: boolean;
  follow_up_ids: string[];
  start_page: number | null;
  end_page: number | null;
  source_section_id: string | null;
  // Enrichment data
  topics: string[];
  investor_intent: string | null;  // clarification, concern, exploration, challenge
  response_posture: string | null;  // confident, cautious, defensive, optimistic, neutral
  boundary_reasoning: string | null;
  confidence: number | null;
}

export interface QAResponse {
  run_id: string;
  qa_units: QAUnit[];
  total_count: number;
  follow_up_count: number;
  unique_questioners: number;
}

// =============================================================================
// Traces
// =============================================================================

export interface EvidenceSpan {
  text: string;
  source: string;
  relevance: string;
}

export interface TraceDecision {
  decision_id: string;
  decision_type: string;
  input_context: string;
  output_decision: string;
  confidence: number | null;
  reasoning: string;
  evidence_spans: EvidenceSpan[];
  timestamp: string | null;
}

export interface StageTrace {
  stage_name: string;
  stage_type: string;
  started_at: string | null;
  completed_at: string | null;
  llm_calls_made: number;
  decisions: TraceDecision[];
  hard_rules_enforced: Record<string, unknown>[];
  warnings: string[];
}

export interface TracesResponse {
  run_id: string;
  stages: StageTrace[];
  total_llm_calls: number;
}

// =============================================================================
// Raw Text
// =============================================================================

export interface PageText {
  page_number: number;
  text: string;
  char_count: number;
}

export interface RawTextResponse {
  run_id: string;
  pages: PageText[];
  total_pages: number;
  total_chars: number;
}

// =============================================================================
// Raw JSON
// =============================================================================

export interface RawJsonResponse {
  run_id: string;
  pipeline_output: Record<string, unknown>;
  stages_output: Record<string, unknown>;
}

// =============================================================================
// Chat
// =============================================================================

export interface ChatCitation {
  type: 'qa' | 'speaker' | 'page';
  ref_id: string;
  label: string;
}

export interface ChatToolCall {
  tool: string;
  params: Record<string, unknown>;
}

export interface ChatMessageItem {
  role: 'user' | 'assistant';
  content: string;
  citations?: ChatCitation[];
  tool_calls?: ChatToolCall[];
  retrieval_source?: string;
  total_time_seconds?: number;
  disclaimer?: string;
}

export interface ChatStreamMetadata {
  tool_calls: ChatToolCall[];
  retrieval_source: string;
}

export interface ChatStreamDone {
  citations: ChatCitation[];
  total_time_seconds: number;
  disclaimer: string;
  model: string;
}

export interface ChatApiResponse {
  run_id: string;
  answer: string;
  citations: ChatCitation[];
  tool_calls: ChatToolCall[];
  retrieval_source: string;
  total_time_seconds: number;
  model: string;
  disclaimer: string;
}
