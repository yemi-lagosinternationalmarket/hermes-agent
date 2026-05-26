import json
from unittest.mock import MagicMock, patch

from tools.dream_tool import _handle_dream


def test_dream_tool_accepts_runtime_kwargs_for_cron_dispatch():
    with patch("agent.dream_engine.DreamEngine") as MockEngine, patch("hermes_state.SessionDB"):
        engine = MagicMock()
        engine.run_dream.return_value = {
            "success": True,
            "insights_count": 0,
            "entries_written": 0,
            "sessions_reviewed": 0,
            "error": "No recent sessions to review",
        }
        MockEngine.return_value = engine

        raw = _handle_dream({"action": "run", "hours": 24, "limit": 20}, task_id="cron-test")

    result = json.loads(raw)
    assert result["status"] == "ok"
    assert result["success"] is True
    engine.run_dream.assert_called_once_with(hours=24, limit=20)
