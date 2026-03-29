"""AI-driven memory management agent."""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import litellm

from .client import KaguraClient
from .exceptions import KaguraAuthError, KaguraConnectionError, KaguraLLMError, KaguraRateLimitError
from .logger import VerboseLogger
from .models import (
    AnalysisResult,
    ExploredMemory,
    LLMUsage,
    Memory,
    MemoryInfo,
    MemoryToStore,
    ProcessResult,
    RecallQuery,
    Session,
)
from .prompts import (
    CONTEXT_SELECTION_PROMPT,
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_analysis_prompt_with_tools,
)

_VALID_HOOK_EVENTS: frozenset[str] = frozenset(
    {"before_response", "after_session", "on_remember", "on_recall"}
)


class KaguraAgent:
    """AI-driven memory management agent using LLM to analyze sessions."""

    def __init__(
        self,
        api_key: str,
        mcp_url: str = "https://memory.kagura-ai.com/mcp",
        model: str = "gpt-5.4-nano",
        context_id: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        llm_api_key: str | None = None,
    ):
        """
        Initialize Kagura Agent.

        Args:
            api_key: Kagura API key
            mcp_url: MCP server URL
            model: LLM model to use (e.g., "gpt-5.4-nano", "claude-sonnet-4-20250514")
            context_id: Default context ID (None or "auto" for auto-selection)
            timeout: Request timeout in seconds
            max_retries: Maximum LLM retry attempts
            llm_api_key: LLM provider API key (passed explicitly, not via env var)
        """
        self.model = model
        self.context_id = context_id
        self.max_retries = max_retries
        self._llm_api_key = llm_api_key
        self.client = KaguraClient(api_key, mcp_url, timeout)
        self.logger: VerboseLogger | None = None

        # Generic cache system with individual timestamps (5 minutes TTL)
        self._cache: dict[str, tuple[Any, float]] = {}
        self._cache_ttl: float = 300.0  # 5 minutes

        # Hooks: event name → list of callables
        self._hooks: dict[str, list[Callable[..., Any]]] = {}
        # Skills: skill name → callable
        self._skills: dict[str, Callable[..., Any]] = {}

    def _is_cache_expired(self, timestamp: float) -> bool:
        """
        Check if cache entry has expired based on TTL.

        Args:
            timestamp: Cache entry timestamp to check

        Returns:
            True if expired, False otherwise
        """
        return time.time() - timestamp > self._cache_ttl

    async def _get_with_cache(
        self,
        cache_key: str,
        fetch_fn: Any,
        label: str,
    ) -> Any:
        """
        Generic cache retrieval helper with individual timestamps.

        Args:
            cache_key: Cache key (e.g., "tools", "contexts")
            fetch_fn: Async function to fetch data if cache miss
            label: Human-readable label for logging

        Returns:
            Cached or freshly fetched data

        Raises:
            KaguraConnectionError: Failed to fetch data
        """
        # Check cache
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if not self._is_cache_expired(timestamp):
                return data

        # Cache miss or expired - fetch new data
        if self.logger:
            self.logger.detail(f"Fetching {label}", "cache miss or expired")

        data = await fetch_fn()
        self._cache[cache_key] = (data, time.time())

        if self.logger and data:
            count = len(data) if isinstance(data, list) else "N/A"
            self.logger.detail(f"Cached {label}", f"{count} items" if count != "N/A" else "")

        return data

    async def _get_tools_with_cache(self) -> list[dict[str, Any]]:
        """
        Get tool definitions with caching.

        Returns:
            List of tool definitions from MCP server

        Raises:
            KaguraConnectionError: Failed to fetch tool definitions
        """
        return await self._get_with_cache(
            cache_key="tools",
            fetch_fn=self.client.get_tool_definitions,
            label="tool definitions",
        )

    async def _get_contexts_with_cache(self) -> list[dict[str, Any]]:
        """
        Get available contexts with caching.

        Returns:
            List of available contexts

        Raises:
            KaguraConnectionError: Failed to fetch contexts
        """

        async def fetch_contexts() -> list[dict[str, Any]]:
            """Wrapper to extract contexts from response."""
            response = await self.client.list_contexts()
            return response.get("contexts", [])

        return await self._get_with_cache(
            cache_key="contexts",
            fetch_fn=fetch_contexts,
            label="contexts",
        )

    async def _build_enhanced_context(self) -> dict[str, Any]:
        """
        Build enhanced context with tool definitions and available contexts.

        Returns:
            Dict containing tools and contexts for LLM prompting

        Raises:
            KaguraConnectionError: Failed to fetch context data
        """
        tools = await self._get_tools_with_cache()
        contexts = await self._get_contexts_with_cache()

        return {"tools": tools, "contexts": contexts}

    def _extract_usage(self, response) -> LLMUsage | None:
        """
        Extract LLM usage from response.

        Args:
            response: LiteLLM response object

        Returns:
            LLMUsage object or None if usage not available
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return None

        return LLMUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            model=self.model,
        )

    async def _call_llm(self, messages: list[dict], temperature: float = 0.3) -> tuple[dict, Any]:
        """
        Call LLM and parse JSON response with retry logic.

        Args:
            messages: Chat messages
            temperature: Sampling temperature

        Returns:
            Tuple of (parsed_data, response_object)

        Raises:
            KaguraLLMError: LLM call failed
            KaguraRateLimitError: Rate limit exceeded
        """
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                    "timeout": 30.0,
                }
                # C-2: Pass LLM API key explicitly instead of relying on env vars
                if self._llm_api_key:
                    kwargs["api_key"] = self._llm_api_key

                response = await litellm.acompletion(**kwargs)

                # Parse response (type: ignore for litellm type issues)
                content = response.choices[0].message.content  # type: ignore
                data = json.loads(content or "{}")

                return data, response

            except litellm.RateLimitError as e:  # pyright: ignore[reportPrivateImportUsage]
                if attempt == self.max_retries - 1:
                    raise KaguraRateLimitError(
                        f"Rate limit exceeded after {self.max_retries} retries"
                    ) from e
                wait_time = 2**attempt
                if self.logger:
                    self.logger.warning(f"Rate limited, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

            except litellm.AuthenticationError as e:  # pyright: ignore[reportPrivateImportUsage]
                raise KaguraLLMError(f"LLM authentication failed: {e}") from e

            except json.JSONDecodeError as e:
                raise KaguraLLMError(f"Invalid JSON response from LLM: {e}") from e

            except litellm.APIError as e:  # pyright: ignore[reportPrivateImportUsage]
                if attempt == self.max_retries - 1:
                    raise KaguraLLMError(
                        f"LLM call failed after {self.max_retries} retries: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

            except Exception as e:
                raise KaguraLLMError(f"Unexpected LLM error: {e}") from e

        raise KaguraLLMError("Max retries exceeded")

    async def _analyze_session(
        self, session: Session, use_enhanced_context: bool = True
    ) -> AnalysisResult:
        """
        Analyze session using LLM.

        Args:
            session: Session to analyze
            use_enhanced_context: Whether to fetch and include tool definitions/contexts
                                  in the prompt (default: True)

        Returns:
            Analysis result with memory operations to perform

        Raises:
            KaguraLLMError: LLM call failed
            KaguraRateLimitError: Rate limit exceeded
        """
        if self.logger:
            self.logger.action("Analyzing session with LLM", f"model={self.model}")

        # Build prompt (with or without enhanced context)
        if use_enhanced_context:
            try:
                enhanced_ctx = await self._build_enhanced_context()
                user_prompt = build_analysis_prompt_with_tools(
                    session, enhanced_ctx["tools"], enhanced_ctx["contexts"]
                )

                if self.logger:
                    tools_count = len(enhanced_ctx["tools"])
                    contexts_count = len(enhanced_ctx["contexts"])
                    self.logger.detail(
                        "Using enhanced context", f"{tools_count} tools, {contexts_count} contexts"
                    )
            except KaguraAuthError:
                raise  # Don't swallow auth errors — surface to caller
            except KaguraConnectionError as e:
                # Fallback to basic prompt only for network/server errors
                if self.logger:
                    self.logger.warning(f"Enhanced context failed, using basic prompt: {e}")
                user_prompt = build_analysis_prompt(session)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Enhanced context failed, using basic prompt: {e}")
                user_prompt = build_analysis_prompt(session)
        else:
            user_prompt = build_analysis_prompt(session)

        if self.logger:
            self.logger.debug(
                "LLM Prompt",
                {"system": SYSTEM_PROMPT[:200] + "...", "user": user_prompt[:500] + "..."},
            )

        # Call LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        analysis_data, response = await self._call_llm(messages, temperature=0.3)

        if self.logger:
            self.logger.debug("LLM Response", analysis_data)

        # Extract and log usage
        llm_usage = self._extract_usage(response)
        if llm_usage and self.logger:
            self.logger.detail(
                "Tokens used",
                f"{llm_usage.total_tokens} (prompt: {llm_usage.prompt_tokens}, "
                f"completion: {llm_usage.completion_tokens})",
            )

        # Create AnalysisResult
        result = AnalysisResult(
            should_remember=analysis_data.get("should_remember", False),
            memories_to_store=analysis_data.get("memories_to_store", []),
            should_recall=analysis_data.get("should_recall", False),
            recall_queries=analysis_data.get("recall_queries", []),
            llm_usage=llm_usage,
        )

        return result

    async def _execute_recalls(
        self, ctx: str, queries: list[RecallQuery], deep: bool, recall_k: int = 5
    ) -> tuple[list[Memory], list[ExploredMemory], list[str]]:
        """
        Execute all recall operations.

        Args:
            ctx: Context ID
            queries: List of recall queries
            deep: Whether to explore related memories
            recall_k: Number of results per query

        Returns:
            Tuple of (recalled_memories, explored_memories, actions)
        """
        recalled: list[Memory] = []
        explored: list[ExploredMemory] = []
        actions: list[str] = []

        for query_info in queries:
            query = query_info.query
            if self.logger:
                self.logger.action("Recalling memories", f'query="{query}"')

            result = await self.client.recall(ctx, query, k=recall_k)

            for mem in result.get("results", []):
                recalled.append(
                    Memory(
                        memory_id=mem["memory_id"],
                        summary=mem["summary"],
                        score=mem.get("score", 0.0),
                    )
                )

            actions.append(f"recall: {query}")
            if self.logger:
                self.logger.detail("Found memories", len(recalled))

            # Deep mode: explore if results found
            if deep and recalled:
                if self.logger:
                    self.logger.action("Deep exploration", f"seed={recalled[0].memory_id}")
                seed_id = recalled[0].memory_id

                try:
                    explore_result = await self.client.explore(
                        ctx, seed_id, depth=2, min_weight=0.0
                    )

                    # Parse explore response: exploration.related_memories
                    exploration = explore_result.get("exploration", {})
                    related = exploration.get("related_memories", [])

                    for rel_mem in related[:5]:
                        explored.append(
                            ExploredMemory(
                                memory_id=rel_mem["memory_id"],
                                summary=rel_mem["summary"],
                                activation=rel_mem.get("activation", 0.0),
                                hop=rel_mem.get("hop", 0),
                            )
                        )

                    actions.append(f"explore: depth=2, found={len(explored)}")
                    if self.logger:
                        self.logger.detail("Explored memories", len(explored))

                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Explore failed: {e}")

        return recalled, explored, actions

    async def _execute_remembers(
        self, ctx: str, memories: list[MemoryToStore]
    ) -> tuple[list[MemoryInfo], list[str]]:
        """
        Execute all remember operations.

        Args:
            ctx: Context ID
            memories: List of memories to store

        Returns:
            Tuple of (remembered_infos, actions)
        """
        remembered: list[MemoryInfo] = []
        actions: list[str] = []

        for mem in memories:
            if self.logger:
                self.logger.action("Storing memory", f'summary="{mem.summary[:50]}..."')

            try:
                result = await self.client.remember(
                    context_id=ctx,
                    summary=mem.summary,
                    content=mem.content,
                    type=mem.type,
                    importance=mem.importance,
                    tags=mem.tags,
                )

                remembered.append(MemoryInfo(memory_id=result["memory_id"], summary=mem.summary))

                actions.append(f"remember: {mem.summary[:50]}")
                if self.logger:
                    self.logger.success(f"Stored memory: {result['memory_id']}")

            except KaguraAuthError:
                raise  # Auth errors are unrecoverable — surface to caller
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to store memory: {e}")

        return remembered, actions

    async def _select_context(self, session: Session) -> str:
        """
        Auto-select best context based on session content.

        Args:
            session: Session to analyze

        Returns:
            Selected context_id

        Raises:
            KaguraLLMError: Context selection failed
        """
        if self.logger:
            self.logger.action("Auto-selecting context")

        # Get available contexts
        contexts_response = await self.client.list_contexts()
        contexts = contexts_response.get("contexts", [])

        if not contexts:
            raise ValueError("No contexts available for auto-selection")

        # If only one context, use it
        if len(contexts) == 1:
            selected = contexts[0]["id"]
            if self.logger:
                self.logger.detail("Auto-selected", f"{contexts[0]['name']} (only option)")
            return selected

        # Use LLM to select best context
        session_summary = "\n".join([f"{m.role}: {m.content[:100]}" for m in session.messages[:3]])

        prompt = CONTEXT_SELECTION_PROMPT.format(
            contexts_json=json.dumps(contexts, indent=2), session_summary=session_summary
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            result, _ = await self._call_llm(messages, temperature=0.2)

            selected_id = result.get("selected_context_id")

            if selected_id and self.logger:
                self.logger.detail("Context selected", result.get("reason", ""))

            return selected_id or contexts[0]["id"]

        except Exception as e:
            # Fallback to first context
            if self.logger:
                self.logger.warning(f"Context selection failed, using first context: {e}")
            return contexts[0]["id"]

    async def process(
        self,
        session: Session,
        context_id: str | None = None,
        deep: bool = False,
        verbose: int = 0,
        recall_k: int = 5,
    ) -> ProcessResult:
        """
        Process session and execute appropriate memory operations.

        The AI analyzes the session and determines what to remember/recall.

        Args:
            session: Conversation session with messages and artifacts
            context_id: Context to use (overrides default, can be "auto")
            deep: Whether to use explore/reference for deeper analysis
            verbose: Log verbosity level 0-3
            recall_k: Number of memories to retrieve per recall query

        Returns:
            ProcessResult with remembered/recalled memories and actions

        Raises:
            KaguraLLMError: LLM call failed
            KaguraAuthError: Authentication failed
            ValueError: context_id is required but not provided
        """
        # Setup logger
        self.logger = VerboseLogger(verbose)

        # Resolve context
        ctx = context_id or self.context_id
        if ctx == "auto":
            ctx = await self._select_context(session)
        elif ctx is None:
            raise ValueError("context_id is required. Set in __init__ or pass to process()")

        self.logger.action("Processing session", f"context={ctx}")

        # Fire before_response hook
        await self._fire_hooks("before_response", session)

        # Analyze session with LLM
        try:
            analysis = await self._analyze_session(session)
        except (KaguraLLMError, KaguraRateLimitError) as e:
            self.logger.error(f"LLM analysis failed: {e}")
            self.logger.warning("Proceeding without AI analysis")
            # Return empty result but don't crash
            return ProcessResult(
                remembered=[],
                recalled=[],
                explored=[],
                context_used=ctx,
                actions=["error: LLM analysis failed"],
                llm_usage=None,
            )

        # Execute memory operations
        recalled, explored, recall_actions = [], [], []
        if analysis.should_recall:
            recalled, explored, recall_actions = await self._execute_recalls(
                ctx, analysis.recall_queries, deep, recall_k
            )
            await self._fire_hooks("on_recall", recalled)

        remembered, remember_actions = [], []
        if analysis.should_remember:
            remembered, remember_actions = await self._execute_remembers(
                ctx, analysis.memories_to_store
            )
            await self._fire_hooks("on_remember", remembered)

        actions = recall_actions + remember_actions

        self.logger.success(
            f"Processing complete: {len(remembered)} remembered, "
            f"{len(recalled)} recalled, {len(explored)} explored"
        )

        result = ProcessResult(
            remembered=remembered,
            recalled=recalled,
            explored=explored,
            context_used=ctx,
            actions=actions,
            llm_usage=analysis.llm_usage,
        )

        # Fire after_session hook
        await self._fire_hooks("after_session", session, result)

        return result

    # ------------------------------------------------------------------
    # Hooks API
    # ------------------------------------------------------------------

    def hook(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Register a hook for a lifecycle event.

        Supported events:
        - ``before_response`` — called before LLM analysis, receives ``(session,)``
        - ``after_session``   — called after process() completes, receives ``(session, result)``
        - ``on_remember``     — called after memories are stored, receives ``(remembered,)``
        - ``on_recall``       — called after memories are retrieved, receives ``(recalled,)``

        Args:
            event: Lifecycle event name.

        Returns:
            Decorator that registers the function as a hook.

        Raises:
            ValueError: If *event* is not a recognised hook event.

        Example::

            @agent.hook("before_response")
            async def auto_recall(session):
                return await agent.client.recall(query=session.messages[-1].content)
        """
        if event not in _VALID_HOOK_EVENTS:
            raise ValueError(
                f"Unknown hook event '{event}'. Valid events: {sorted(_VALID_HOOK_EVENTS)}"
            )

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._hooks.setdefault(event, []).append(fn)
            return fn

        return decorator

    async def _fire_hooks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """
        Invoke all hooks registered for *event* sequentially.

        Both coroutine functions and plain callables are supported.
        """
        for fn in self._hooks.get(event, []):
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Skills API
    # ------------------------------------------------------------------

    def skill(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Register a named skill.

        Args:
            name: Skill name used to invoke it later.

        Returns:
            Decorator that registers the function as a skill.

        Example::

            @agent.skill("summarize")
            async def summarize_context(context_id):
                memories = await agent.client.recall(query="*", context_id=context_id)
                return memories
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._skills[name] = fn
            return fn

        return decorator

    async def run_skill(self, skill_name: str, **kwargs: Any) -> Any:
        """
        Invoke a registered skill by name.

        Args:
            skill_name: Skill name to run.
            **kwargs: Keyword arguments forwarded to the skill function.

        Returns:
            Whatever the skill function returns.

        Raises:
            ValueError: If no skill with *skill_name* is registered.
        """
        if skill_name not in self._skills:
            available = sorted(self._skills)
            raise ValueError(f"Skill '{skill_name}' not found. Available skills: {available}")
        result = self._skills[skill_name](**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def list_skills(self) -> list[str]:
        """Return a sorted list of registered skill names."""
        return sorted(self._skills)

    async def close(self):
        """Close underlying client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
