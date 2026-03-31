"""Prompt templates for LLM-based session analysis."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Session

# Display/formatting constants (DRY: avoid magic numbers)
MAX_CONTEXTS_DISPLAY = 5  # Maximum contexts to show in prompt
CONTEXT_ID_DISPLAY_LENGTH = 8  # Context ID prefix length for readability
MAX_SUMMARY_LENGTH = 100  # Maximum context summary display length
MAX_MESSAGES = 20  # Maximum session messages to include
MAX_ARTIFACTS = 5  # Maximum artifacts to include
ARTIFACT_PREVIEW_LENGTH = 500  # Artifact content preview length

SYSTEM_PROMPT = """\
You are a memory extraction expert for Kagura Memory Cloud. \
Your role is to analyze conversation sessions and output structured JSON \
that determines what to store (REMEMBER) and search (RECALL).

<guidelines>
<remember_when>
- A decision or architectural choice was made
- A solution, pattern, or best practice was discovered
- Technical knowledge was shared (code, config, commands)
- A bug was fixed and the root cause identified
- The user explicitly asks to remember something
</remember_when>

<recall_when>
- The user asks a question or mentions a topic — ALWAYS recall to check for relevant memories
- A problem is mentioned that may have been solved before
- Context from previous work or discussions would help
- When in doubt, set should_recall to true — searching is cheap, missing context is costly
</recall_when>
</guidelines>

<quality_standards>
<summary_rules>
Write the CONCLUSION, not the process. 10-250 characters.
Use the same language as the conversation (Japanese session = Japanese summary).
</summary_rules>

<types>
Use one of: code, note, decision, bug-fix, feature, learning
</types>

<importance_scale>
0.9-1.0: Security fixes, breaking changes, critical architecture decisions
0.7-0.8: Reusable patterns, important configurations, key learnings
0.5-0.6: General notes, minor decisions, useful reference material
0.3-0.4: Low-priority observations, temporary context
</importance_scale>

<tags_rules>
Include category tags and entity tags for searchability.
- Category: "category:{domain}" (e.g., "category:料理", "category:backend")
- Entity: key terms with writing variations for recall
- Japanese: include kanji, katakana, hiragana (e.g., ["鯖", "サバ", "さば"])
- Okurigana variants (e.g., ["引越し", "引っ越し", "ひっこし"])
- Tech: abbreviated and full forms (e.g., ["DB", "データベース", "database"])
</tags_rules>

<recall_query_rules>
Write queries with both semantic meaning and specific keywords (hybrid search).
Use filters.tags to narrow results when specific topics are mentioned.
Example: query="認証エラーの対処法", filters={"tags": ["auth"]}
</recall_query_rules>
</quality_standards>

<examples>
<example>
<input>User learns FastAPI uses Depends() for dependency injection</input>
<output>
{
  "should_remember": true,
  "memories_to_store": [
    {
      "summary": "FastAPI DI: inject via Depends(get_db) in function args",
      "content": "FastAPI DI pattern: use Depends(factory) to inject services. Override in tests.",
      "type": "learning",
      "importance": 0.7,
      "tags": ["backend", "fastapi", "python", "dependency-injection"]
    }
  ]
}
</output>
</example>

<example>
<input>User asks about OAuth2 implementation they worked on before</input>
<output>
{
  "should_recall": true,
  "recall_queries": [
    {
      "query": "OAuth2 implementation token refresh authentication",
      "reason": "User asks about previous OAuth2 work"
    }
  ]
}
</output>
</example>
</examples>

<output_format>
Return ONLY valid JSON matching this schema. No markdown, no explanation, no preamble.
{
  "should_remember": boolean,
  "memories_to_store": [
    {
      "summary": "string (10-250 chars, conclusion in session language)",
      "content": "string (full context and details in session language)",
      "type": "code|note|decision|bug-fix|feature|learning",
      "importance": 0.0-1.0,
      "tags": ["string"]
    }
  ],
  "should_recall": boolean,
  "recall_queries": [
    {
      "query": "string (semantic + keyword hybrid search query)",
      "reason": "string",
      "filters": {"tags": ["string"]} or null
    }
  ]
}
</output_format>"""


def _format_session_content(session: "Session") -> tuple[str, str]:
    """
    Format session messages and artifacts (DRY: used by both prompt builders).

    Args:
        session: Session with messages and artifacts

    Returns:
        Tuple of (formatted_messages, formatted_artifacts)
    """
    # Format messages (limit to recent messages to avoid token bloat)
    messages_text = "\n".join(
        [f"[{msg.role.upper()}]: {msg.content}" for msg in session.messages[-MAX_MESSAGES:]]
    )

    # Format artifacts if present
    artifacts_text = ""
    if session.artifacts:
        artifacts_text = "\n\n## Attached Artifacts:\n"
        for i, art in enumerate(session.artifacts[:MAX_ARTIFACTS], 1):
            content_preview = (
                art.content[:ARTIFACT_PREVIEW_LENGTH] + "..."
                if len(art.content) > ARTIFACT_PREVIEW_LENGTH
                else art.content
            )
            artifacts_text += f"\n[{i}] Type: {art.type}"
            if art.source:
                artifacts_text += f" | Source: {art.source}"
            if art.language:
                artifacts_text += f" | Language: {art.language}"
            artifacts_text += f"\n{content_preview}\n"

    return messages_text, artifacts_text


def build_analysis_prompt(session: "Session") -> str:
    """
    Build LLM prompt for session analysis (basic version).

    Args:
        session: Session with messages and artifacts

    Returns:
        Formatted prompt string
    """
    messages_text, artifacts_text = _format_session_content(session)

    prompt = f"""<conversation>
{messages_text}
{artifacts_text}
</conversation>

<instructions>
Analyze the conversation above and determine what memory operations to perform.
Focus on technical knowledge, patterns, decisions, and solutions.
Return JSON response only.
</instructions>"""

    return prompt


CONTEXT_SELECTION_PROMPT = """\
Select the most appropriate context for this conversation.

<contexts>
{contexts_json}
</contexts>

<conversation>
{session_summary}
</conversation>

<criteria>
1. Topic alignment: match the conversation subject to the context name and summary
2. Recency: prefer recently active contexts over dormant ones
3. Scope: prefer specific contexts over general ones
</criteria>

Return JSON only:
{{
  "selected_context_id": "uuid-string",
  "reason": "brief explanation"
}}"""


def _format_tool_for_prompt(tool: dict[str, Any]) -> str:
    """
    Format full tool schema for LLM prompt.

    Passes complete MCP tool descriptions to the LLM without truncation,
    so the LLM receives the same guidance that direct MCP clients get
    (e.g., search optimization hints, good/bad examples).

    Args:
        tool: Tool definition dict with name, description, inputSchema

    Returns:
        Formatted tool description string
    """
    name = tool.get("name", "unknown")
    description = tool.get("description", "")
    input_schema = tool.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    required_params = "\n".join(
        f"    - {p} ({properties[p].get('type', 'any')}): {properties[p].get('description', '')}"
        for p in required
        if p in properties
    )

    result = f"### {name}\n{description}\n"
    if required_params:
        result += f"\n  Required:\n{required_params}\n"
    return result


def build_analysis_prompt_with_tools(
    session: "Session", tools: list[dict[str, Any]], contexts: list[dict[str, Any]]
) -> str:
    """
    Build LLM prompt with dynamic tool definitions and available contexts.

    This enhanced version includes actual tool specifications from the MCP server,
    allowing the LLM to understand exact parameter names and types.

    Args:
        session: Session with messages and artifacts
        tools: Tool definitions from MCP server (tools/list)
        contexts: Available contexts from list_contexts

    Returns:
        Formatted prompt string with tool definitions
    """
    # Format available tools
    tools_desc = "".join([_format_tool_for_prompt(tool) for tool in tools])

    # Format available contexts (limit to most recent)
    contexts_desc = ""
    for ctx in contexts[:MAX_CONTEXTS_DISPLAY]:
        ctx_name = ctx.get("name", "Unknown")
        ctx_id = ctx.get("id", "")[:CONTEXT_ID_DISPLAY_LENGTH]  # Short ID for readability
        ctx_summary = (ctx.get("summary") or "No description")[:MAX_SUMMARY_LENGTH]
        contexts_desc += f"  - {ctx_name} (id: {ctx_id}...): {ctx_summary}\n"

    # Reuse session formatting logic (DRY)
    messages_text, artifacts_text = _format_session_content(session)

    prompt = f"""<available_tools>
{tools_desc}
</available_tools>

<available_contexts>
{contexts_desc}
</available_contexts>

<conversation>
{messages_text}
{artifacts_text}
</conversation>

<instructions>
Analyze the conversation above and determine what memory operations to perform.
You have access to the tools and contexts listed above.
Focus on technical knowledge, patterns, decisions, and solutions.
Return JSON response only.
</instructions>"""

    return prompt
