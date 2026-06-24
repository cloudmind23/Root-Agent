"""
Shared pytest configuration and fixtures for the Root Agent test suite.

Fixtures defined here are available to all test modules without explicit import.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models import AgentRun, AgentStatus, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# pytest-asyncio: use asyncio mode for all tests
# ---------------------------------------------------------------------------

# pyproject.toml or pytest.ini sets asyncio_mode = "auto"; this marker
# ensures compatibility when running via plain `pytest` as well.


# ---------------------------------------------------------------------------
# AgentRun factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_agent_run():
    """
    Factory fixture: returns a callable that creates AgentRun instances.

    Usage::

        def test_something(make_agent_run):
            run = make_agent_run(namespace="prod", cluster="prod-us-east-1")
    """

    def _factory(**kwargs) -> AgentRun:
        return AgentRun(
            run_id=kwargs.pop("run_id", str(uuid.uuid4())),
            alert_summary=kwargs.pop(
                "alert_summary",
                "Pod api-server-0 in production is CrashLoopBackOff",
            ),
            **kwargs,
        )

    return _factory


@pytest.fixture
def pending_run(make_agent_run) -> AgentRun:
    """A freshly created AgentRun in PENDING state."""
    return make_agent_run()


@pytest.fixture
def running_run(make_agent_run) -> AgentRun:
    """An AgentRun in RUNNING state."""
    run = make_agent_run()
    run.set_status(AgentStatus.RUNNING)
    return run


@pytest.fixture
def awaiting_run(make_agent_run) -> AgentRun:
    """An AgentRun in AWAITING_APPROVAL state with a pending fix."""
    run = make_agent_run(namespace="production", cluster="prod-us-east-1")
    run.set_status(AgentStatus.AWAITING_APPROVAL)
    run.pending_fix = {
        "action": "patch",
        "resource_type": "deployment",
        "resource_name": "api-server",
        "namespace": "production",
        "patch_body": {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "api-server", "resources": {"limits": {"memory": "512Mi"}}}
                        ]
                    }
                }
            }
        },
        "human_approved": False,
    }
    return run


@pytest.fixture
def completed_run(make_agent_run) -> AgentRun:
    """An AgentRun in COMPLETED state with a diagnosis."""
    run = make_agent_run()
    run.set_status(AgentStatus.RUNNING)
    run.llm_steps = 3
    run.diagnosis = "Container OOMKilled due to memory limit of 256Mi. Increased to 512Mi."
    run.fix_applied = "Patched deployment api-server memory limit to 512Mi."
    run.set_status(AgentStatus.COMPLETED)
    return run


# ---------------------------------------------------------------------------
# ToolCall / ToolResult factories
# ---------------------------------------------------------------------------


@pytest.fixture
def make_tool_call():
    """Factory for ToolCall instances."""

    def _factory(**kwargs) -> ToolCall:
        return ToolCall(
            tool_use_id=kwargs.pop("tool_use_id", f"toolu_{uuid.uuid4().hex[:8]}"),
            name=kwargs.pop("name", "get_pod_logs"),
            input=kwargs.pop("input", {"pod_name": "api-0", "namespace": "prod"}),
            **kwargs,
        )

    return _factory


@pytest.fixture
def make_tool_result():
    """Factory for ToolResult instances."""

    def _factory(**kwargs) -> ToolResult:
        return ToolResult(
            tool_use_id=kwargs.pop("tool_use_id", f"toolu_{uuid.uuid4().hex[:8]}"),
            name=kwargs.pop("name", "get_pod_logs"),
            output=kwargs.pop("output", "Container logs: OOMKilled"),
            **kwargs,
        )

    return _factory


# ---------------------------------------------------------------------------
# Kubernetes client mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_core_v1():
    """Mock kubernetes.client.CoreV1Api with sensible defaults."""
    core = MagicMock()
    core.read_namespaced_pod_log.return_value = "INFO starting\nERROR OOMKilled"
    core.list_namespaced_event.return_value.items = []
    core.list_event_for_all_namespaces.return_value.items = []
    core.delete_namespaced_pod.return_value = None
    return core


@pytest.fixture
def mock_apps_v1():
    """Mock kubernetes.client.AppsV1Api with sensible defaults."""
    apps = MagicMock()
    apps.patch_namespaced_deployment.return_value = None
    apps.patch_namespaced_stateful_set.return_value = None
    apps.patch_namespaced_daemon_set.return_value = None
    return apps


@pytest.fixture
def mock_k8s_clients(mock_core_v1, mock_apps_v1):
    """Patch get_k8s_clients to return (mock_core_v1, mock_apps_v1)."""
    import app.tools.executor as executor_module  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    with patch.object(
        executor_module,
        "get_k8s_clients",
        return_value=(mock_core_v1, mock_apps_v1),
    ):
        yield mock_core_v1, mock_apps_v1


# ---------------------------------------------------------------------------
# Anthropic API mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_text_response():
    """A mock Anthropic Messages response with a single text block (end_turn)."""
    block = MagicMock()
    block.type = "text"
    block.text = "Root cause: OOMKilled. Recommendation: increase memory limit."

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


@pytest.fixture
def mock_anthropic_tool_response():
    """
    Factory fixture: returns a callable that builds a mock Anthropic tool_use response.

    Usage::

        def test_something(mock_anthropic_tool_response):
            response = mock_anthropic_tool_response(
                tool_name="get_pod_logs",
                tool_id="toolu_01",
                tool_input={"pod_name": "api-0", "namespace": "prod"},
            )
    """

    def _factory(
        tool_name: str = "get_pod_logs",
        tool_id: str = "toolu_01",
        tool_input: dict[str, Any] | None = None,
    ) -> MagicMock:
        if tool_input is None:
            tool_input = {"pod_name": "api-0", "namespace": "prod"}

        block = MagicMock()
        block.type = "tool_use"
        block.id = tool_id
        block.name = tool_name
        block.input = tool_input

        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [block]
        return response

    return _factory


# ---------------------------------------------------------------------------
# Redis mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis_store():
    """
    In-memory Redis substitute. Returns a tuple of (AsyncMock, store_dict, history_list).

    - store_dict:   {key: json_string} — mirrors Redis key-value storage
    - history_list: [(score, run_id)] — mirrors the sorted set, newest-first
    """
    store: dict[str, str] = {}
    history: list[tuple[float, str]] = []

    redis = AsyncMock()

    async def _set(key: str, value: str, ex=None):
        store[key] = value

    async def _get(key: str) -> str | None:
        return store.get(key)

    async def _zadd(name: str, mapping: dict):
        for run_id, score in mapping.items():
            history.append((score, run_id))
        history.sort(key=lambda x: x[0], reverse=True)

    async def _zrevrange(name: str, start: int, end: int) -> list[str]:
        return [rid for _, rid in history[start : end + 1]]

    async def _ping():
        return True

    redis.set = AsyncMock(side_effect=_set)
    redis.get = AsyncMock(side_effect=_get)
    redis.zadd = AsyncMock(side_effect=_zadd)
    redis.zrevrange = AsyncMock(side_effect=_zrevrange)
    redis.ping = AsyncMock(side_effect=_ping)

    return redis, store, history
