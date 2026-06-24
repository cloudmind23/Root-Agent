"""
Tests for app/agent/graph.py

Strategy
--------
The Anthropic client and tool executor are mocked so no real API calls or
cluster access is needed. LangGraph nodes are tested as plain async functions
(not through the compiled graph) to keep tests fast and deterministic.

Covers
------
llm_node
  - calls Anthropic API with correct model/tools/messages
  - appends assistant message to state
  - increments iteration counter and run.llm_steps
  - handles tool_use and text content blocks

tool_node
  - dispatches all tool_use blocks in parallel
  - appends tool_result user message
  - records ToolCall / ToolResult on run
  - intercepts apply_fix without approval → sets pending_fix
  - tool exception → is_error=True, loop continues

approval_node
  - sets run.pending_fix from state
  - transitions run to AWAITING_APPROVAL

route_after_llm
  - tool_use stop_reason → "tool_node"
  - end_turn stop_reason → END
  - max iterations → END
  - pending_fix present → "approval_node"

route_after_tools
  - no pending_fix → "llm_node"
  - pending_fix present → "approval_node"

run_graph
  - happy path: PENDING run → COMPLETED with diagnosis
  - run with apply_fix intercept → AWAITING_APPROVAL
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graph import (
    AgentState,
    approval_node,
    llm_node,
    route_after_llm,
    route_after_tools,
    tool_node,
)
from app.models import AgentRun, AgentStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(**kwargs) -> AgentRun:
    return AgentRun(
        run_id=str(uuid.uuid4()),
        alert_summary=kwargs.pop("alert_summary", "Pod is CrashLoopBackOff in production"),
        status=AgentStatus.RUNNING,
        **kwargs,
    )


def _make_state(run: AgentRun | None = None, **kwargs) -> AgentState:
    return AgentState(
        run=run or _make_run(),
        messages=kwargs.pop("messages", [{"role": "user", "content": "investigate"}]),
        iterations=kwargs.pop("iterations", 0),
        stop_reason=kwargs.pop("stop_reason", None),
        pending_fix=kwargs.pop("pending_fix", None),
        **kwargs,
    )


def _mock_anthropic_response(
    stop_reason: str = "end_turn",
    content_blocks: list | None = None,
) -> MagicMock:
    """Return a mock Anthropic Messages response."""
    if content_blocks is None:
        content_blocks = [
            MagicMock(type="text", text="Root cause: OOMKilled. Recommend increasing memory limit."),
        ]
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content_blocks
    return response


def _tool_use_block(name: str, tool_id: str, input_: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_
    return block


# ---------------------------------------------------------------------------
# llm_node
# ---------------------------------------------------------------------------


class TestLlmNode:
    @pytest.mark.asyncio
    async def test_calls_anthropic_and_appends_message(self):
        state = _make_state()
        response = _mock_anthropic_response()

        with patch("app.agent.graph.anthropic.AsyncAnthropic") as mock_cls, \
             patch("app.agent.graph.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-test"
            mock_settings.return_value.anthropic_model = "claude-sonnet-4-6"
            mock_settings.return_value.anthropic_max_tokens = 4096
            mock_settings.return_value.anthropic_max_iterations = 15

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_client

            result = await llm_node(state)

        # Assistant message appended
        assert result["messages"][-1]["role"] == "assistant"
        # Iteration counter incremented
        assert result["iterations"] == 1
        assert result["run"].llm_steps == 1
        # stop_reason recorded
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_use_block_serialised_correctly(self):
        state = _make_state()
        tool_block = _tool_use_block("get_pod_logs", "toolu_01", {"pod_name": "api-0", "namespace": "prod"})
        response = _mock_anthropic_response(stop_reason="tool_use", content_blocks=[tool_block])

        with patch("app.agent.graph.anthropic.AsyncAnthropic") as mock_cls, \
             patch("app.agent.graph.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-test"
            mock_settings.return_value.anthropic_model = "claude-sonnet-4-6"
            mock_settings.return_value.anthropic_max_tokens = 4096
            mock_settings.return_value.anthropic_max_iterations = 15

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_client

            result = await llm_node(state)

        assistant_content = result["messages"][-1]["content"]
        tool_use_blocks = [b for b in assistant_content if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_pod_logs"
        assert tool_use_blocks[0]["id"] == "toolu_01"

    @pytest.mark.asyncio
    async def test_passes_all_tools_to_api(self):
        state = _make_state()
        response = _mock_anthropic_response()

        with patch("app.agent.graph.anthropic.AsyncAnthropic") as mock_cls, \
             patch("app.agent.graph.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-test"
            mock_settings.return_value.anthropic_model = "claude-sonnet-4-6"
            mock_settings.return_value.anthropic_max_tokens = 4096
            mock_settings.return_value.anthropic_max_iterations = 15

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_client

            await llm_node(state)

        _, kwargs = mock_client.messages.create.call_args
        tool_names = [t["name"] for t in kwargs["tools"]]
        for expected in ["get_pod_logs", "apply_fix", "notify_slack", "web_search"]:
            assert expected in tool_names


# ---------------------------------------------------------------------------
# tool_node
# ---------------------------------------------------------------------------


class TestToolNode:
    def _state_with_tool_use(self, tool_name: str, tool_id: str, input_: dict) -> AgentState:
        state = _make_state()
        state["messages"].append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tool_id, "name": tool_name, "input": input_}
            ],
        })
        return state

    @pytest.mark.asyncio
    async def test_dispatches_tool_and_appends_result(self):
        state = self._state_with_tool_use("get_pod_logs", "toolu_01", {"pod_name": "api-0", "namespace": "prod"})

        with patch("app.agent.graph.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = "OOMKilled logs here"
            result = await tool_node(state)

        # tool_result appended as user message
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        tool_results = [b for b in last_msg["content"] if b["type"] == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_01"
        assert "OOMKilled" in tool_results[0]["content"]

    @pytest.mark.asyncio
    async def test_records_tool_call_and_result_on_run(self):
        state = self._state_with_tool_use("get_pod_logs", "toolu_01", {"pod_name": "api-0", "namespace": "prod"})

        with patch("app.agent.graph.dispatch", new_callable=AsyncMock, return_value="logs"):
            result = await tool_node(state)

        run: AgentRun = result["run"]
        assert len(run.tool_calls) == 1
        assert run.tool_calls[0].name == "get_pod_logs"
        assert len(run.tool_results) == 1

    @pytest.mark.asyncio
    async def test_tool_exception_sets_is_error(self):
        state = self._state_with_tool_use("get_pod_logs", "toolu_01", {"pod_name": "api-0", "namespace": "prod"})

        with patch("app.agent.graph.dispatch", new_callable=AsyncMock, side_effect=Exception("k8s down")):
            result = await tool_node(state)

        last_msg = result["messages"][-1]
        tr_block = last_msg["content"][0]
        assert tr_block["is_error"] is True

    @pytest.mark.asyncio
    async def test_apply_fix_without_approval_intercepted(self):
        """apply_fix with human_approved=False should NOT call dispatch — sets pending_fix instead."""
        state = self._state_with_tool_use(
            "apply_fix",
            "toolu_02",
            {"action": "restart", "resource_type": "deployment", "resource_name": "api", "human_approved": False},
        )

        with patch("app.agent.graph.dispatch", new_callable=AsyncMock) as mock_dispatch:
            result = await tool_node(state)

        # dispatch should NOT have been called for apply_fix
        mock_dispatch.assert_not_awaited()
        # pending_fix set
        assert result["pending_fix"] is not None
        assert result["pending_fix"]["action"] == "restart"
        # tool_result still appended (with pending_approval status)
        last_msg = result["messages"][-1]
        tr_block = last_msg["content"][0]
        assert "pending_approval" in tr_block["content"]

    @pytest.mark.asyncio
    async def test_multiple_tools_dispatched_in_parallel(self):
        state = _make_state()
        state["messages"].append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "get_pod_logs",
                 "input": {"pod_name": "api-0", "namespace": "prod"}},
                {"type": "tool_use", "id": "t2", "name": "get_events",
                 "input": {}},
            ],
        })

        with patch("app.agent.graph.dispatch", new_callable=AsyncMock, return_value="output"):
            result = await tool_node(state)

        last_msg = result["messages"][-1]
        assert len(last_msg["content"]) == 2

    @pytest.mark.asyncio
    async def test_no_tool_use_blocks_is_noop(self):
        state = _make_state()
        state["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
        original_msg_count = len(state["messages"])

        result = await tool_node(state)
        # No new user message appended (noop)
        assert len(result["messages"]) == original_msg_count


# ---------------------------------------------------------------------------
# approval_node
# ---------------------------------------------------------------------------


class TestApprovalNode:
    @pytest.mark.asyncio
    async def test_sets_pending_fix_and_status(self):
        fix = {"action": "patch", "resource_type": "deployment", "resource_name": "api", "human_approved": False}
        state = _make_state(pending_fix=fix)

        result = await approval_node(state)

        run: AgentRun = result["run"]
        assert run.status == AgentStatus.AWAITING_APPROVAL
        assert run.pending_fix == fix

    @pytest.mark.asyncio
    async def test_pending_fix_none_allowed(self):
        state = _make_state(pending_fix=None)
        result = await approval_node(state)
        assert result["run"].status == AgentStatus.AWAITING_APPROVAL
        assert result["run"].pending_fix is None


# ---------------------------------------------------------------------------
# route_after_llm
# ---------------------------------------------------------------------------


class TestRouteAfterLlm:
    def test_tool_use_routes_to_tool_node(self):
        state = _make_state(stop_reason="tool_use", iterations=1)
        assert route_after_llm(state) == "tool_node"

    def test_end_turn_routes_to_end(self):
        state = _make_state(stop_reason="end_turn", iterations=1)
        from langgraph.graph import END
        assert route_after_llm(state) == END

    def test_max_iterations_routes_to_end(self):
        with patch("app.agent.graph.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_max_iterations = 5
            state = _make_state(stop_reason="tool_use", iterations=5)
            from langgraph.graph import END
            assert route_after_llm(state) == END

    def test_pending_fix_routes_to_approval(self):
        fix = {"action": "restart"}
        state = _make_state(stop_reason="tool_use", iterations=1, pending_fix=fix)
        assert route_after_llm(state) == "approval_node"


# ---------------------------------------------------------------------------
# route_after_tools
# ---------------------------------------------------------------------------


class TestRouteAfterTools:
    def test_no_pending_fix_routes_to_llm(self):
        state = _make_state(pending_fix=None)
        assert route_after_tools(state) == "llm_node"

    def test_pending_fix_routes_to_approval(self):
        state = _make_state(pending_fix={"action": "restart"})
        assert route_after_tools(state) == "approval_node"


# ---------------------------------------------------------------------------
# run_graph (integration-style, graph nodes mocked)
# ---------------------------------------------------------------------------


class TestRunGraph:
    @pytest.mark.asyncio
    async def test_completed_run_has_diagnosis(self):
        """
        Simulate a single llm_node → END loop where Claude returns end_turn.
        The graph should exit COMPLETED with the assistant text as diagnosis.
        """
        from app.agent.graph import run_graph

        run = _make_run()

        final_state = AgentState(
            run=run,
            messages=[
                {"role": "user", "content": "investigate"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Root cause: OOMKilled. Fix: increase memory."}],
                },
            ],
            iterations=1,
            stop_reason="end_turn",
            pending_fix=None,
        )
        final_state["run"].llm_steps = 1

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)

        with patch("app.agent.graph.get_graph", return_value=mock_graph):
            result = await run_graph(run)

        assert result.status == AgentStatus.COMPLETED
        assert "OOMKilled" in result.diagnosis

    @pytest.mark.asyncio
    async def test_awaiting_approval_run_not_overwritten(self):
        """
        If the graph exits with AWAITING_APPROVAL, run_graph must not
        overwrite the status to COMPLETED.
        """
        from app.agent.graph import run_graph

        run = _make_run()

        final_state = AgentState(
            run=run,
            messages=[{"role": "user", "content": "investigate"}],
            iterations=2,
            stop_reason=None,
            pending_fix={"action": "restart"},
        )
        final_state["run"].set_status(AgentStatus.AWAITING_APPROVAL)
        final_state["run"].pending_fix = {"action": "restart"}

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)

        with patch("app.agent.graph.get_graph", return_value=mock_graph):
            result = await run_graph(run)

        assert result.status == AgentStatus.AWAITING_APPROVAL
