"""Prompt templates for the 2-phase chatbot agent.

Phase 1: Tool selection — model outputs JSON action
Phase 2: Grounded synthesis — model answers ONLY from evidence

All prompts enforce strict grounding: every claim must cite [qa_XXX] or [page_N].
Hallucination guardrails are embedded in every synthesis prompt.
"""


def build_tool_selection_prompt(
    company: str,
    quarter: str,
    year: int,
    speaker_count: int,
    qa_count: int,
    speaker_names: str,
    user_question: str,
) -> str:
    """Build the Phase 1 prompt that asks the LLM to select a retrieval tool.

    The model must output ONLY a JSON object with action + tool + params.
    """
    return f"""You select which data retrieval tool to use for a user's question about a {company} {quarter} {year} earnings call.

DATA AVAILABLE:
- {speaker_count} speakers: {speaker_names}
- {qa_count} Q&A exchanges with questioner names, responder names, question text, response text
- Searchable by: keyword, questioner_name, responder_name, is_follow_up

TOOL PRIORITY RULES:
1. For questions about specific analysts or speakers: use search_qa_units with questioner_name or search_speakers with name_query
2. For topic/keyword questions: use search_qa_units with keyword
3. For management/role questions: use search_speakers with role
4. For call metadata: use get_run_metadata
5. For follow-up chains: use get_follow_up_chain
6. search_full_text is a LAST RESORT only when QA search cannot match the question
7. NEVER guess speaker roles. Use the structured data.

Output ONLY a JSON object. Choose one action:

{{"action": "structured_search", "tool": "search_qa_units", "params": {{"keyword": "text"}}}}
{{"action": "structured_search", "tool": "search_qa_units", "params": {{"questioner_name": "name"}}}}
{{"action": "structured_search", "tool": "search_qa_units", "params": {{"keyword": "text", "limit": 10}}}}
{{"action": "structured_search", "tool": "search_speakers", "params": {{"role": "management"}}}}
{{"action": "structured_search", "tool": "search_speakers", "params": {{"name_query": "name"}}}}
{{"action": "structured_search", "tool": "get_run_metadata", "params": {{}}}}
{{"action": "structured_search", "tool": "search_full_text", "params": {{"keyword": "text"}}}}
{{"action": "structured_search", "tool": "search_strategic_statements", "params": {{"keyword": "text"}}}}
{{"action": "structured_search", "tool": "get_follow_up_chain", "params": {{"qa_id": "qa_000"}}}}

Question: {user_question}
Action:"""


def build_synthesis_prompt(
    company: str,
    quarter: str,
    year: int,
    user_question: str,
    evidence: str,
) -> str:
    """Build the Phase 2 prompt that synthesizes an answer from evidence.

    Strict grounding rules prevent hallucination.
    """
    return f"""You are answering a question about a {company} {quarter} {year} earnings call.
You MUST answer ONLY using the evidence provided below. Do not add any information that is not in the evidence.

EVIDENCE:
{evidence}

RULES:
1. Every factual claim MUST cite a source: [qa_XXX] or [page_N]
2. Do NOT invent, assume, or infer facts not present in the evidence
3. Do NOT use your general knowledge about the company or industry
4. If the evidence does not contain enough information to answer, say:
   "The available data does not contain sufficient information to answer this question."
5. Quote relevant text directly when possible using quotation marks
6. Keep the answer concise and factual
7. If you are uncertain about an inference, prefix it with "Based on the evidence,"

STRICTLY FORBIDDEN:
- Do NOT invent regulations, metrics, percentages, or financial figures not in the evidence
- Do NOT fabricate strategic initiatives or business plans
- Do NOT guess analyst intent or sentiment unless explicitly stated
- Do NOT add context from your training data about the company
- Do NOT list raw Q&A pairs as the answer. Synthesize and analyze instead.

Use markdown: **bold** for key terms, bullet points for structure.

Question: {user_question}
Answer:"""


def build_vector_fallback_prompt(
    company: str,
    quarter: str,
    year: int,
    user_question: str,
    evidence: str,
) -> str:
    """Build synthesis prompt specifically for vector search results.

    Adds extra caution since vector results may be less precisely matched.
    """
    return f"""You are answering a question about a {company} {quarter} {year} earnings call.
The following evidence was found via semantic search (it may not be an exact match for the question).
Answer ONLY using what is explicitly stated in the evidence.

EVIDENCE:
{evidence}

RULES:
1. Every factual claim MUST cite a source: [qa_XXX] or [page_N]
2. Do NOT invent, assume, or infer facts not present in the evidence
3. Do NOT use your general knowledge about the company or industry
4. If the evidence is only loosely related to the question, say so explicitly
5. If the evidence does not contain enough information, say:
   "The available data does not contain sufficient information to answer this question."
6. Quote relevant text directly when possible

STRICTLY FORBIDDEN:
- Do NOT invent regulations, metrics, percentages, or financial figures not in the evidence
- Do NOT fabricate strategic initiatives or business plans
- Do NOT guess analyst intent or sentiment unless explicitly stated
- Do NOT add context from your training data about the company

Use markdown: **bold** for key terms, bullet points for structure.

Question: {user_question}
Answer:"""


def build_summary_synthesis_prompt(
    company: str,
    quarter: str,
    year: int,
    user_question: str,
    evidence: str,
) -> str:
    """Build a synthesis prompt for summary/overview requests.

    Instructs the LLM to identify themes across Q&A, NOT dump individual pairs.
    """
    return f"""You are summarizing the Q&A session of a {company} {quarter} {year} earnings call.
You MUST use ONLY the evidence provided below. Do not add any information not in the evidence.

EVIDENCE (all Q&A exchanges):
{evidence}

YOUR TASK:
Produce a THEMATIC SUMMARY, not a list of individual Q&A pairs.

1. Identify the 3-6 MAJOR THEMES or topics discussed across all Q&A exchanges
2. For each theme:
   - Name the theme clearly (e.g., "**Margins & Profitability**", "**Growth Strategy**", "**Regulatory Risks**")
   - Describe what analysts were trying to understand
   - Summarize how management responded
   - Cite all relevant Q&A exchanges: [qa_XXX]
3. Note patterns: What did analysts focus on most? Were there recurring concerns?
4. Keep it analytical and high-level

CRITICAL RULES:
- This is a THEMATIC SUMMARY, NOT a question-by-question walkthrough
- Do NOT list each Q&A pair individually
- Do NOT copy full questions or answers verbatim
- Group related questions across different analysts into themes
- Every claim MUST cite at least one source: [qa_XXX]
- Do NOT invent or infer topics not present in the evidence
- Do NOT use your general knowledge about the company
- If insufficient evidence exists for a theme, say so explicitly

FORMAT: Use markdown with **bold** theme names and bullet points for structure.

Question: {user_question}
Answer:"""


def build_analyst_synthesis_prompt(
    company: str,
    quarter: str,
    year: int,
    analyst_name: str,
    user_question: str,
    evidence: str,
) -> str:
    """Build a synthesis prompt for analyst-specific questions.

    Uses structured QA data to describe what a specific analyst asked about.
    """
    return f"""You are answering a question about what {analyst_name} asked during a {company} {quarter} {year} earnings call Q&A session.
You MUST use ONLY the evidence provided below.

EVIDENCE (Q&A exchanges involving {analyst_name}):
{evidence}

YOUR TASK:
1. List the topics/areas {analyst_name} asked about
2. For each question or topic:
   - Briefly describe what {analyst_name} was asking
   - Summarize how management responded
   - Note if there were follow-up questions
   - Cite the Q&A exchange: [qa_XXX]
3. If {analyst_name} had a clear area of focus or concern, highlight that

RULES:
- Every claim MUST cite a source: [qa_XXX]
- Use the questioner names from the evidence as SOURCE OF TRUTH for who asked what
- Do NOT guess analyst intent beyond what is explicitly asked
- Do NOT invent topics or concerns not in the evidence
- If the evidence does not contain Q&A exchanges from {analyst_name}, say so clearly
- Do NOT use your general knowledge about the company

FORMAT: Use markdown with **bold** for topic areas and bullet points.

Question: {user_question}
Answer:"""


# Predefined suggested questions based on data summary
def build_suggested_questions(qa_count: int, speaker_count: int) -> list[str]:
    """Generate suggested starter questions for the chat UI."""
    return [
        "Who are the management speakers in this call?",
        "What were the key themes in the Q&A session?",
        f"Summarize the main concerns raised by analysts (from {qa_count} Q&A exchanges)",
        "What guidance or forward-looking statements were made?",
    ]
