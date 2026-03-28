"""Prompt templates for LLM-based session analysis."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Session

# Display/formatting constants (DRY: avoid magic numbers)
MAX_CONTEXTS_DISPLAY = 5  # Maximum contexts to show in prompt
CONTEXT_ID_DISPLAY_LENGTH = 8  # Context ID prefix length for readability
MAX_SUMMARY_LENGTH = 100  # Maximum context summary display length
MAX_DESCRIPTION_LENGTH = 200  # Maximum tool description length
MAX_OPTIONAL_PARAMS = 3  # Maximum optional parameters to display
MAX_MESSAGES = 20  # Maximum session messages to include
MAX_ARTIFACTS = 5  # Maximum artifacts to include
ARTIFACT_PREVIEW_LENGTH = 500  # Artifact content preview length

SYSTEM_PROMPT = """You are a memory extraction expert for Kagura Memory Cloud.

Your role is to analyze conversation sessions and determine:
1. What information should be remembered (stored as memories)
2. What information should be recalled (searched from past memories)

Guidelines for REMEMBER:
- Important decisions or choices made
- Learned patterns, solutions, or best practices
- Technical knowledge (code snippets, configurations)
- Facts to preserve for future reference
- User explicitly requests "remember this"

Guidelines for RECALL:
- User asks a question that might benefit from past context
- Problem mentioned that could have been solved before
- Reference to previous work or discussions
- Need for background information or examples

Memory Quality Guidelines:
- Summary: 100-250 characters, focus on conclusions not process
- Type: code|note|decision|learning|pattern
- Importance: 0.9-1.0 (critical), 0.7-0.8 (important), 0.5-0.6 (useful), 0.3-0.4 (reference)
- Tags: Include domain tags (backend, frontend) and tech tags (fastapi, react, etc.)

Output Format: Return ONLY valid JSON matching this exact schema:
{
  "should_remember": boolean,
  "memories_to_store": [
    {
      "summary": "string (100-250 chars, reusable conclusion)",
      "content": "string (full content with context)",
      "type": "code|note|decision|learning|pattern",
      "importance": 0.0-1.0,
      "tags": ["tag1", "tag2", "tag3"]
    }
  ],
  "should_recall": boolean,
  "recall_queries": [
    {
      "query": "string (semantic search query)",
      "reason": "why this recall is relevant"
    }
  ]
}"""


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

    prompt = f"""Analyze this conversation session and determine what memory operations to perform:

## Conversation:
{messages_text}
{artifacts_text}

## Instructions:
1. Determine if anything valuable should be REMEMBERED (stored permanently)
2. Determine if past memories should be RECALLED (searched) to provide context
3. For memories to store:
   - Extract the core learning/decision (not just "we discussed X")
   - Write clear, reusable summaries (100-250 chars)
   - Assign appropriate type and importance
   - Add relevant tags for future search
4. For recall queries:
   - Write semantic search queries (not just keywords)
   - Explain why the recall would be helpful

Focus on technical knowledge, patterns, decisions, and solutions.
Avoid remembering casual conversation, greetings, or temporary information.

Return JSON response only (no markdown, no explanation):"""

    return prompt


CONTEXT_SELECTION_PROMPT = """Given these available contexts and a conversation session, \
select the most appropriate context.

## Available Contexts:
{contexts_json}

## Conversation Summary:
{session_summary}

## Selection Criteria:
1. Topic alignment with context summary/name
2. Recent usage (prefer active contexts)
3. Workspace vs personal scope

Return JSON only:
{{
  "selected_context_id": "uuid-string",
  "reason": "brief explanation why this context was selected"
}}"""


def _format_tool_parameter(name: str, schema: dict[str, Any]) -> str:
    """
    Format a tool parameter for display in prompt.

    Args:
        name: Parameter name
        schema: Parameter schema dict with type and description

    Returns:
        Formatted parameter string
    """
    param_type = schema.get("type", "any")
    description = schema.get("description", "")
    return f"    - {name} ({param_type}): {description}"


def _summarize_tool_schema(tool: dict[str, Any]) -> str:
    """
    Summarize tool schema for prompt (reduce token usage).

    Args:
        tool: Tool definition dict with name, description, inputSchema

    Returns:
        Summarized tool description string
    """
    name = tool.get("name", "unknown")
    # First line only, truncate to max length
    description = tool.get("description", "").split("\n")[0][:MAX_DESCRIPTION_LENGTH]
    input_schema = tool.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Format required parameters
    required_params = []
    for param_name in required:
        if param_name in properties:
            required_params.append(_format_tool_parameter(param_name, properties[param_name]))

    # Format optional parameters (limit to most important ones)
    optional_params = []
    for param_name, param_schema in list(properties.items())[:MAX_OPTIONAL_PARAMS]:
        if param_name not in required:
            optional_params.append(_format_tool_parameter(param_name, param_schema))

    result = f"• {name}: {description}\n"
    if required_params:
        result += "  Required:\n" + "\n".join(required_params) + "\n"
    if optional_params:
        result += "  Optional:\n" + "\n".join(optional_params) + "\n"

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
    tools_desc = "".join([_summarize_tool_schema(tool) for tool in tools])

    # Format available contexts (limit to most recent)
    contexts_desc = ""
    for ctx in contexts[:MAX_CONTEXTS_DISPLAY]:
        ctx_name = ctx.get("name", "Unknown")
        ctx_id = ctx.get("id", "")[:CONTEXT_ID_DISPLAY_LENGTH]  # Short ID for readability
        ctx_summary = (ctx.get("summary") or "No description")[:MAX_SUMMARY_LENGTH]
        contexts_desc += f"  - {ctx_name} (id: {ctx_id}...): {ctx_summary}\n"

    # Reuse session formatting logic (DRY)
    messages_text, artifacts_text = _format_session_content(session)

    prompt = f"""Analyze this conversation session and determine what memory operations to perform.

## Available Kagura Memory Tools:
{tools_desc}

## Available Contexts:
{contexts_desc}

## Conversation:
{messages_text}
{artifacts_text}

## Instructions:
1. Determine if anything valuable should be REMEMBERED (stored permanently)
2. Determine if past memories should be RECALLED (searched) to provide context
3. For memories to store:
   - Extract the core learning/decision (not just "we discussed X")
   - Write clear, reusable summaries (100-250 chars)
   - Assign appropriate type and importance
   - Add relevant tags for future search
4. For recall queries:
   - Write semantic search queries (not just keywords)
   - Explain why the recall would be helpful

Focus on technical knowledge, patterns, decisions, and solutions.
Avoid remembering casual conversation, greetings, or temporary information.

Return JSON response only (no markdown, no explanation):"""

    return prompt
