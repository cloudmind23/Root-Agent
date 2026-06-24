"""
Tests for app/main.py — FastAPI routes

Strategy
--------
- httpx.AsyncClient is used as the test HTTP client against the app
- Redis is mocked at the module level so no real Redis is needed
- The agent background task (_run_agent_loop) is patched to a no-op so
  tests don't spin up a real LangGraph loop

Covers
------
POST /agent/analyze
  - returns 202 with a PENDING AgentRun
  - alert_summary too short → 422
  - run is persisted to Redis

GET /agent/runs/{run_id}
  - returns the run for a known ID
  - returns 404 for an unknown ID

POST /agent/approve/{run_id}
  - approved=True resumes the agent loop and returns the run
  - approved=False cancels the run
  - 409 when run is not in AWAITING_APPROVAL state

GET /agent/history
  - returns a list of AgentRunSummary
  - filters by status
  - filters by cluster
  - respects limit/offset

GET /healthz
GET /readyz
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import AgentRun, AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(**kwargs) -> AgentRun:
    return AgentRun(
        run_id=kwargs.pop("run_id", str(uuid.uuid4())),
        alert_summary=kwargs.pop("alert_summary", "Pod is CrashLoopBackOff in production"),
        **kwargs,
    )


def _serialise(run: AgentRun) -> str:
    return run.model_dump_json()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """
    Patch app.main._redis with an AsyncMock that stores data in-memory.
    """
    store: dict[str, str] = {}
    history: list[tuple[float, str]] = []  # [(score, run_id)]

    redis = AsyncMock()

    async def _set(key: str, value: str, ex=None):
        store[key] = value

    async def _get(key: str):
        return store.get(key)

    async def _zadd(name: str, mapping: dict):
        for run_id, score in mapping.items():
            history.append((score, run_id))
        history.sort(key=lambda x: x[0], reverse=True)

    async def _zrevrange(name: str, start: int, end: int):
        return [rid for _, rid in history[start : end + 1]]

    async def _ping():
        return True

    redis.set = AsyncMock(side_effect=_set)
    redis.get = AsyncMock(side_effect=_get)
    redis.zadd = AsyncMock(side_effect=_zadd)
    redis.zrevrange = AsyncMock(side_effect=_zrevrange)
    redis.ping = AsyncMock(side_effect=_ping)

    with patch("app.main._redis", redis):
        yield redis, store, history


@pytest_asyncio.fixture
async def client():
    """httpx AsyncClient wired to the FastAPI app (no real server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# POST /agent/analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_returns_202_with_pending_run(self, client, mock_redis):
        with patch("app.main._run_agent_loop", new_callable=AsyncMock), \
             patch("asyncio.create_task"):
            response = await client.post(
                "/agent/analyze",
                json={"alert_summary": "Pod api-0 is CrashLoopBackOff in production"},
            )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        assert "run_id" in data
        assert data["alert_summary"] == "Pod api-0 is CrashLoopBackOff in production"

    @pytest.mark.asyncio
    async def test_run_persisted_to_redis(self, client, mock_redis):
        _, store, _ = mock_redis
        with patch("asyncio.create_task"):
            response = await client.post(
                "/agent/analyze",
                json={"alert_summary": "Pod api-0 is CrashLoopBackOff in production"},
            )
        run_id = response.json()["run_id"]
        assert any(run_id in k for k in store)

    @pytest.mark.asyncio
    async def test_with_namespace_and_cluster(self, client, mock_redis):
        with patch("asyncio.create_task"):
            response = await client.post(
                "/agent/analyze",
                json={
                    "alert_summary": "Pod api-0 is CrashLoopBackOff in production",
                    "namespace": "production",
                    "cluster": "prod-us-east-1",
                },
            )
        data = response.json()
        assert data["namespace"] == "production"
        assert data["cluster"] == "prod-us-east-1"

    @pytest.mark.asyncio
    async def test_short_alert_summary_returns_422(self, client, mock_redis):
        response = await client.post(
            "/agent/analyze",
            json={"alert_summary": "too short"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /agent/runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRun:
    @pytest.mark.asyncio
    async def test_returns_run_for_known_id(self, client, mock_redis):
        _, store, _ = mock_redis
        run = _make_run(status=AgentStatus.RUNNING)
        store[f"root-agent:run:{run.run_id}"] = _serialise(run)

        response = await client.get(f"/agent/runs/{run.run_id}")
        assert response.status_code == 200
        assert response.json()["run_id"] == run.run_id
        assert response.json()["status"] == "running"

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_id(self, client, mock_redis):
        response = await client.get("/agent/runs/does-not-exist")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_run_with_tool_calls_included(self, client, mock_redis):
        _, store, _ = mock_redis
        run = _make_run()
        from app.models import ToolCall
        run.add_tool_call(ToolCall(tool_use_id="t1", name="get_pod_logs", input={"pod_name": "api-0", "namespace": "prod"}))
        store[f"root-agent:run:{run.run_id}"] = _serialise(run)

        response = await client.get(f"/agent/runs/{run.run_id}")
        assert len(response.json()["tool_calls"]) == 1
        assert response.json()["tool_calls"][0]["name"] == "get_pod_logs"


# ---------------------------------------------------------------------------
# POST /agent/approve/{run_id}
# ---------------------------------------------------------------------------


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_true_resumes_loop(self, client, mock_redis):
        _, store, _ = mock_redis
        run = _make_run(status=AgentStatus.AWAITING_APPROVAL)
        run.pending_fix = {"action": "restart", "resource_type": "deployment", "resource_name": "api", "human_approved": False}
        store[f"root-agent:run:{run.run_id}"] = _serialise(run)

        with patch("asyncio.create_task") as mock_task:
            response = await client.post(
                f"/agent/approve/{run.run_id}",
                json={"approved": True, "approver": "alice@company.com"},
            )

        assert response.status_code == 200
        mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_false_cancels_run(self, client, mock_redis):
        _, store, _ = mock_redis
        run = _make_run(status=AgentStatus.AWAITING_APPROVAL)
        run.pending_fix = {"action": "restart", "resource_type": "deployment", "resource_name": "api"}
        store[f"root-agent:run:{run.run_id}"] = _serialise(run)

        response = await client.post(
            f"/agent/approve/{run.run_id}",
            json={"approved": False, "approver": "bob@company.com"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_approve_non_awaiting_run_returns_409(self, client, mock_redis):
        _, store, _ = mock_redis
        run = _make_run(status=AgentStatus.RUNNING)
        store[f"root-agent:run:{run.run_id}"] = _serialise(run)

        response = await client.post(
            f"/agent/approve/{run.run_id}",
            json={"approved": True},
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_approve_unknown_run_returns_404(self, client, mock_redis):
        response = await client.post(
            "/agent/approve/no-such-run",
            json={"approved": True},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /agent/history
# ---------------------------------------------------------------------------


class TestHistory:
    def _seed(self, store, history_list, runs: list[AgentRun]):
        """Helper: populate the in-memory mock store and history index."""
        for run in runs:
            store[f"root-agent:run:{run.run_id}"] = _serialise(run)
            history_list.append((run.created_at.timestamp(), run.run_id))
        history_list.sort(key=lambda x: x[0], reverse=True)

    @pytest.mark.asyncio
    async def test_returns_list(self, client, mock_redis):
        _, store, history_list = mock_redis
        run = _make_run()
        self._seed(store, history_list, [run])

        response = await client.get("/agent/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert any(r["run_id"] == run.run_id for r in response.json())

    @pytest.mark.asyncio
    async def test_empty_history(self, client, mock_redis):
        response = await client.get("/agent/history")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_filter_by_status(self, client, mock_redis):
        _, store, history_list = mock_redis
        completed = _make_run()
        completed.set_status(AgentStatus.COMPLETED)
        failed = _make_run()
        failed.set_status(AgentStatus.FAILED)
        self._seed(store, history_list, [completed, failed])

        response = await client.get("/agent/history?status=completed")
        data = response.json()
        assert all(r["status"] == "completed" for r in data)
        assert any(r["run_id"] == completed.run_id for r in data)

    @pytest.mark.asyncio
    async def test_filter_by_cluster(self, client, mock_redis):
        _, store, history_list = mock_redis
        run_a = _make_run(cluster="prod-us-east-1")
        run_b = _make_run(cluster="staging-eu-west-1")
        self._seed(store, history_list, [run_a, run_b])

        response = await client.get("/agent/history?cluster=prod-us-east-1")
        data = response.json()
        assert all(r["cluster"] == "prod-us-east-1" for r in data)

    @pytest.mark.asyncio
    async def test_limit_param(self, client, mock_redis):
        _, store, history_list = mock_redis
        runs = [_make_run() for _ in range(5)]
        self._seed(store, history_list, runs)

        response = await client.get("/agent/history?limit=2")
        assert len(response.json()) <= 2

    @pytest.mark.asyncio
    async def test_limit_above_100_returns_422(self, client, mock_redis):
        response = await client.get("/agent/history?limit=200")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------


class TestHealthProbes:
    @pytest.mark.asyncio
    async def test_healthz_returns_200(self, client, mock_redis):
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_readyz_returns_200_when_redis_up(self, client, mock_redis):
        response = await client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readyz_returns_503_when_redis_down(self, client):
        with patch("app.main._redis", None):
            response = await client.get("/readyz")
        assert response.status_code == 503
