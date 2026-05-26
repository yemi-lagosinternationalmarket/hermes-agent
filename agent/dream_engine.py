"""DreamEngine — session review and memory consolidation.

Periodically "dreams" through recent sessions to extract durable insights
and consolidate them into both local memory files (MEMORY.md / USER.md)
and the Hindsight cloud memory (if configured).

Inspired by how sleep consolidates memories in biological systems: the agent
reviews its recent conversations, identifies patterns and lessons learned,
and writes the most valuable ones to long-term memory.

Architecture:
    DreamEngine.gather_sessions()  — query SessionDB for recent sessions
    DreamEngine.extract_insights() — build a review prompt from sessions
    DreamEngine.run_dream()        — execute dream via AIAgent with memory toolset
    DreamEngine.apply_consolidation() — parse and write insights to memory files + Hindsight

Integration points:
    - tools/dream_tool.py — user-facing 'dream' tool (run, config, history)
    - cron/scheduler.py   — dream-type jobs route through DreamEngine
    - cron/jobs.py         — dream jobs auto-set enabled_toolsets=['memory']
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Dream history file — tracks past dream runs for the 'history' action.
_DREAM_HISTORY_FILE = get_hermes_home() / "cron" / "dream_history.json"

# Maximum characters of session content to include in the review prompt.
# Keeps the dream prompt under the model's context window.
_MAX_SESSION_CONTENT_CHARS = 4000

# Maximum number of user/assistant message pairs to extract per session.
_MAX_MESSAGES_PER_SESSION = 20


class DreamEngine:
    """Session review and memory consolidation engine.

    Gathers recent session transcripts, builds a review prompt, runs an
    AIAgent with the memory toolset to extract insights, and applies
    the consolidation results to MEMORY.md / USER.md.
    """

    def __init__(self, session_db=None):
        """Initialize the DreamEngine.

        Args:
            session_db: Optional SessionDB instance. If None, a new one is
                        created lazily on first use.
        """
        self._session_db = session_db

    def _get_session_db(self):
        """Return the SessionDB, creating one if needed."""
        if self._session_db is None:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        return self._session_db

    def gather_sessions(
        self,
        hours: int = 24,
        limit: int = 20,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query SessionDB for recent sessions within the time window.

        Args:
            hours: How many hours back to look (default 24).
            limit: Maximum number of sessions to return (default 20).
            exclude_sources: Session sources to exclude (e.g. ['cron']).

        Returns:
            List of session dicts with id, source, title, message_count,
            and a 'messages' key containing the conversation transcript.
        """
        db = self._get_session_db()
        cutoff_ts = time.time() - (hours * 3600)

        # Use search_sessions to get recent sessions, then filter by time.
        sessions = db.search_sessions(limit=limit * 2)  # over-fetch for filtering

        recent = []
        for s in sessions:
            # Filter by time window
            started_at = s.get("started_at", 0)
            if started_at < cutoff_ts:
                continue

            # Filter out excluded sources
            source = s.get("source", "")
            if exclude_sources and source in exclude_sources:
                continue

            # Skip cron sessions (they're the dream itself or other scheduled work)
            if source == "cron":
                continue

            session_id = s.get("id", "")
            if not session_id:
                continue

            # Fetch messages for this session
            try:
                messages = db.get_messages(session_id)
            except Exception as e:
                logger.debug("DreamEngine: failed to get messages for %s: %s", session_id, e)
                messages = []

            # Build a compact transcript
            transcript = self._compact_transcript(messages)
            if not transcript:
                continue

            recent.append({
                "id": session_id,
                "source": source,
                "title": s.get("title", ""),
                "message_count": s.get("message_count", 0),
                "started_at": started_at,
                "transcript": transcript,
            })

            if len(recent) >= limit:
                break

        logger.info("DreamEngine: gathered %d sessions from the last %d hours", len(recent), hours)
        return recent

    def _compact_transcript(self, messages: List[Dict[str, Any]]) -> str:
        """Build a compact text transcript from session messages.

        Extracts only user and assistant messages (skipping tool calls/results),
        limited to _MAX_MESSAGES_PER_SESSION pairs and _MAX_SESSION_CONTENT_CHARS
        total characters.
        """
        pairs = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if not content:
                continue
            # Skip system messages and tool results
            if role in ("system", "tool"):
                continue
            # Truncate individual messages
            if len(content) > 500:
                content = content[:500] + "..."
            pairs.append(f"[{role}]: {content}")

            if len(pairs) >= _MAX_MESSAGES_PER_SESSION:
                break

        transcript = "\n".join(pairs)
        if len(transcript) > _MAX_SESSION_CONTENT_CHARS:
            transcript = transcript[:_MAX_SESSION_CONTENT_CHARS] + "\n...[truncated]"

        return transcript

    def extract_insights(self, sessions: List[Dict[str, Any]]) -> str:
        """Build the dream review prompt from gathered sessions.

        Args:
            sessions: List of session dicts from gather_sessions().

        Returns:
            A prompt string for the dream agent to analyze.
        """
        if not sessions:
            return ""

        session_blocks = []
        for i, s in enumerate(sessions, 1):
            title = s.get("title", "Untitled") or "Untitled"
            source = s.get("source", "unknown")
            msg_count = s.get("message_count", 0)
            transcript = s.get("transcript", "")

            block = (
                f"### Session {i}: {title}\n"
                f"Source: {source} | Messages: {msg_count}\n"
                f"```\n{transcript}\n```\n"
            )
            session_blocks.append(block)

        sessions_text = "\n".join(session_blocks)

        prompt = (
            "You are in a memory consolidation cycle (\"dreaming\"). Your task is to\n"
            "review recent sessions and extract durable insights worth remembering.\n\n"
            "ANALYZE these session transcripts and identify:\n"
            "1. **User preferences** — communication style, habits, pet peeves, "
            "recurring requests\n"
            "2. **Environment facts** — tools, configs, project structures, "
            "OS details discovered\n"
            "3. **Lessons learned** — mistakes to avoid, successful patterns, "
            "workflow optimizations\n"
            "4. **Corrections** — times the user corrected you or said 'remember this'\n\n"
            "For EACH insight worth saving, output it as a JSON object in a JSON array:\n"
            "```json\n"
            "[\n"
            '  {"target": "memory", "content": "insight text here"},\n'
            '  {"target": "user", "content": "user preference here"}\n'
            "]\n"
            "```\n\n"
            "RULES:\n"
            "- Only include genuinely useful, non-obvious insights\n"
            "- Skip trivial facts, temporary task state, or things easily re-discovered\n"
            "- Each entry should be concise (1-2 sentences max)\n"
            "- 'target': 'memory' for environment/technical notes, 'user' for user profile\n"
            "- If nothing worth saving, output an empty array: []\n"
            "- Do NOT duplicate existing memory entries\n\n"
            f"## Recent Sessions ({len(sessions)} sessions)\n\n"
            f"{sessions_text}\n"
        )

        return prompt

    def run_dream(self, agent=None, sessions: Optional[List[Dict[str, Any]]] = None,
                  hours: int = 24, limit: int = 20) -> Dict[str, Any]:
        """Execute a dream cycle: gather sessions, run agent, apply results.

        Args:
            agent: Optional pre-configured AIAgent. If None, creates one with
                   the memory toolset.
            sessions: Optional pre-gathered sessions. If None, gathers them.
            hours: Look-back window for gathering sessions.
            limit: Max sessions to gather.

        Returns:
            Dict with keys: success, insights_count, entries_written, error.
        """
        result = {
            "success": False,
            "insights_count": 0,
            "entries_written": 0,
            "sessions_reviewed": 0,
            "error": None,
        }

        try:
            # Gather sessions
            if sessions is None:
                sessions = self.gather_sessions(hours=hours, limit=limit)

            result["sessions_reviewed"] = len(sessions)

            if not sessions:
                result["success"] = True
                result["error"] = "No recent sessions to review"
                self._record_history(result)
                return result

            # Build the dream prompt
            dream_prompt = self.extract_insights(sessions)
            if not dream_prompt:
                result["success"] = True
                result["error"] = "No content to analyze"
                self._record_history(result)
                return result

            # Run the dream agent
            if agent is None:
                agent = self._create_dream_agent()

            logger.info("DreamEngine: running dream with %d sessions", len(sessions))

            # Use run_conversation for the full interface
            agent_result = agent.run_conversation(dream_prompt)
            response = agent_result.get("final_response", "") if isinstance(agent_result, dict) else str(agent_result)

            # Parse insights from the response
            entries = self._parse_insights(response)
            result["insights_count"] = len(entries)

            # Apply consolidation
            if entries:
                written = self.apply_consolidation(entries)
                result["entries_written"] = written

            result["success"] = True
            logger.info(
                "DreamEngine: dream complete — %d insights found, %d entries written",
                result["insights_count"], result["entries_written"],
            )

        except Exception as e:
            logger.exception("DreamEngine: dream failed: %s", e)
            result["error"] = str(e)

        self._record_history(result)
        return result

    def _create_dream_agent(self):
        """Create an AIAgent configured for dreaming (memory toolset only)."""
        from run_agent import AIAgent

        # Load config for model/provider resolution
        cfg = {}
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            pass

        model = ""
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str):
            model = model_cfg
        elif isinstance(model_cfg, dict):
            model = model_cfg.get("default", "")

        # Resolve provider
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider()

        agent = AIAgent(
            model=model or runtime.get("model", ""),
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            enabled_toolsets=["memory"],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            platform="cron",
            session_id=f"dream_{int(time.time())}",
        )

        return agent

    def _parse_insights(self, response: str) -> List[Dict[str, str]]:
        """Parse the dream agent's response into insight entries.

        Looks for JSON arrays in the response. Handles markdown code fences.

        Returns:
            List of dicts with 'target' and 'content' keys.
        """
        entries = []

        # Try to find JSON array in the response
        text = response.strip()

        # Strip markdown code fences if present
        if "```" in text:
            import re
            # Find the last JSON block (the final answer, not examples)
            blocks = re.findall(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
            for block in reversed(blocks):
                try:
                    parsed = json.loads(block.strip())
                    if isinstance(parsed, list):
                        entries = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # If no fenced block found, try parsing the whole response
        if not entries:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    entries = parsed
            except json.JSONDecodeError:
                pass

        # Validate entries
        valid = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            target = entry.get("target", "").strip()
            content = entry.get("content", "").strip()
            if target in ("memory", "user") and content:
                valid.append({"target": target, "content": content})

        return valid

    def apply_consolidation(self, entries: List[Dict[str, str]]) -> int:
        """Write dream insights to MEMORY.md/USER.md and Hindsight cloud.

        Writes to both local memory files (for system prompt injection) and
        Hindsight (for long-term cloud recall/reflect). Tagged with
        source:dream so dream-originated memories are identifiable.

        Args:
            entries: List of dicts with 'target' and 'content' keys.

        Returns:
            Number of entries successfully written to local memory.
        """
        from tools.memory_tool import MemoryStore

        store = MemoryStore()
        store.load_from_disk()

        written = 0
        for entry in entries:
            target = entry["target"]
            content = entry["content"]

            try:
                result = store.add(target, content)
                if result.get("success"):
                    written += 1
                    logger.debug("DreamEngine: wrote to %s: %s", target, content[:80])
                else:
                    error = result.get("error", "unknown")
                    if "already exists" in str(error).lower():
                        logger.debug("DreamEngine: skipping duplicate in %s", target)
                    else:
                        logger.warning("DreamEngine: failed to write to %s: %s", target, error)
            except Exception as e:
                logger.warning("DreamEngine: error writing to %s: %s", target, e)

        self._retain_to_hindsight(entries)
        return written

    def _retain_to_hindsight(self, entries: List[Dict[str, str]]) -> None:
        """Retain dream insights to Hindsight cloud memory."""
        if not entries:
            return

        try:
            config = self._load_hindsight_config()
            if not config:
                logger.debug("DreamEngine: no Hindsight config found, skipping cloud retain")
                return

            api_key = config.get("apiKey", "")
            bank_id = config.get("bank_id", "hermes")
            if not api_key:
                logger.debug("DreamEngine: no Hindsight API key, skipping cloud retain")
                return

            from hindsight_client import Hindsight
            client = Hindsight(
                base_url=config.get("api_url", "https://api.hindsight.vectorize.io"),
                api_key=api_key,
            )

            content = "Dream consolidation insights:\n" + "\n".join(
                f"- [{e['target']}] {e['content']}" for e in entries
            )

            from hermes_time import now as _hermes_now
            _run_sync = self._get_hindsight_run_sync()
            _run_sync(client.aretain(
                bank_id=bank_id,
                content=content,
                context="nightly dream cycle — consolidated insights from recent sessions",
                tags=["source:dream"],
                metadata={
                    "source": "dream_engine",
                    "timestamp": _hermes_now().isoformat(),
                    "insights_count": str(len(entries)),
                },
            ))
            logger.info("DreamEngine: retained %d insights to Hindsight", len(entries))

        except ImportError:
            logger.debug("DreamEngine: hindsight_client not installed, skipping cloud retain")
        except Exception as e:
            logger.warning("DreamEngine: failed to retain to Hindsight: %s", e)

    @staticmethod
    def _load_hindsight_config() -> Optional[dict]:
        """Load Hindsight config from the standard location."""
        config_path = get_hermes_home() / "hindsight" / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _get_hindsight_run_sync():
        """Get the async-to-sync bridge for Hindsight calls."""
        import asyncio
        def run_sync(coro, timeout=30):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, coro)
                        return future.result(timeout=timeout)
                else:
                    return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
            except RuntimeError:
                return asyncio.run(coro)
        return run_sync

    def _record_history(self, result: Dict[str, Any]) -> None:
        """Record a dream run in the history file for the 'history' action."""
        try:
            _DREAM_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

            history = []
            if _DREAM_HISTORY_FILE.exists():
                try:
                    with open(_DREAM_HISTORY_FILE, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except (json.JSONDecodeError, OSError):
                    history = []

            from hermes_time import now as _hermes_now
            entry = {
                "timestamp": _hermes_now().isoformat(),
                "success": result.get("success", False),
                "sessions_reviewed": result.get("sessions_reviewed", 0),
                "insights_count": result.get("insights_count", 0),
                "entries_written": result.get("entries_written", 0),
                "error": result.get("error"),
            }
            history.append(entry)

            # Keep only last 50 entries
            if len(history) > 50:
                history = history[-50:]

            with open(_DREAM_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.debug("DreamEngine: failed to record history: %s", e)

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent dream run history.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of dream run records.
        """
        if not _DREAM_HISTORY_FILE.exists():
            return []

        try:
            with open(_DREAM_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            return history[-limit:]
        except (json.JSONDecodeError, OSError):
            return []

    def get_config(self) -> Dict[str, Any]:
        """Return current dreaming configuration from config.yaml.

        Returns:
            Dict with dreaming config keys and their values.
        """
        cfg = {}
        try:
            from hermes_cli.config import load_config
            full_cfg = load_config() or {}
            cfg = full_cfg.get("dreaming", {})
        except Exception:
            pass

        return {
            "enabled": cfg.get("enabled", True),
            "schedule": cfg.get("schedule", "0 3 * * *"),
            "hours": cfg.get("hours", 24),
            "limit": cfg.get("limit", 20),
            "deliver": cfg.get("deliver", "local"),
        }

    def build_dream_prompt(self, job: dict) -> str:
        """Build the effective prompt for a dream-type cron job.

        Called by cron/scheduler.py when it detects job type == 'dream'.
        This replaces the normal _build_job_prompt path for dream jobs.

        Args:
            job: The cron job dict.

        Returns:
            The dream prompt string.
        """
        hours = job.get("dream_hours", 24)
        limit = job.get("dream_limit", 20)

        sessions = self.gather_sessions(hours=hours, limit=limit)

        if not sessions:
            return (
                "[SILENT] Dream cycle: no recent sessions to review. "
                "Nothing to consolidate."
            )

        return self.extract_insights(sessions)
