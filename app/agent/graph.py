"""
LangGraph agent loop for Root Agent.

Graph topology
--------------

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────────┐
    │  llm_node    │  ← call Claude with current message history + tools
    └────┬─────────┘
         │
    ┌────▼─────────┐        ┌──────────────────┐
    │  route_node  │───────►│ approval_node    │
    └────┬─────────┘        └──────┬───────────┘
         │ (tool_calls)            │ (pending fix persisted;
         │                         │  status → AWAITING_APPROVAL)
    ┌────▼──────────┐              │
    │  tool_node    │              ▼
    └────┬──────────┘          ┌──────┐
         │                     │  END │
         └─────────────────────┤      │
              (loop back)      └──────┘
         │ (stop_reason=end_turn
         │  or max_iterations)
         ▼
      ┌──────┐
      │  END │
      └──────┘

Nodes
-----
llm_node        — call the Anthropic API; append assistant message to state
route_node      — inspect stop_reason; decide next node (tool_node / approval_node / END)
tool_node       — execute all tool_use blocks in parallel; append tool_results
approval_node   — persist pending fix to AgentRun; set AWAITING_APPROVAL status
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional, TypedDict

import anthropic
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.models import AgentRun, AgentStatus, ToolCall, ToolResult
from app.tools.definitions import ALL_TOOLS
from app.tools.executor import dispatch

logger = logging.getLogger("root_agent.graph")

# ---------------------------------------------------------------------------
# Graph state schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Typed state dict that flows through every LangGraph node.

    Fields
    ------
    run         : AgentRun — the persistent run record being updated in-place
    messages    : list[dict] — Anthropic API message history (user/assistant turns)
    iterations  : int — number of Claude API calls made so far
    stop_reason : str | None — last stop_reason from Claude ('tool_use', 'end_turn', …)
    pending_fix : dict | None — apply_fix payload awaiting human approval
    """

    run: AgentRun
    messages: list[dict]
    iterations: int
    stop_reason: Optional[str]
    pending_fix: Optional[dict[str, Any]]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Root Agent, an expert autonomous Kubernetes Site Reliability Engineer agent.

Your goal is to diagnose and resolve Kubernetes cluster issues with minimal human intervention.
You have access to a set of tools that let you inspect pods, read logs, search runbooks,
and apply approved fixes.

## Process

1. **Investigate first.** Always start by fetching logs (get_pod_logs) and cluster events
   (get_events) before drawing conclusions.
2. **Describe resources** (describe_resource) when you need full spec/status detail —
   especially for scheduling failures or probe issues.
3. **Search runbooks** (search_runbooks) to check if a known fix exists before inventing one.
4. **Web search** (web_search) only as a last resort for unknown error codes or third-party
   controller errors.
5. **Propose fixes conservatively.** When you identify a root cause and a safe fix, call
   apply_fix with human_approved=false — you MUST wait for the operator to approve it.
   Never set human_approved=true yourself.
6. **Notify** (notify_slack) at key milestones: when you need human approval, when a fix
   has been applied, or when you need escalation.
7. **Conclude** with a clear summary: root cause, action taken (or proposed), and next steps.

## Constraints

- Never apply a fix without human approval.
- Do not loop more than 15 iterations — conclude with partial findings if needed.
- Keep Slack messages concise and action-oriented.
- If a tool call fails, try an alternative approach before giving up.
"""


# ---------------------------------------------------------------------------
# Node: llm_node
# ---------------------------------------------------------------------------


async def llm_node(state: AgentState) -> AgentState:
    """
    Invoke the Claude API with the current message history.

    Appends the assistant's response to state['messages'] and updates
    state['stop_reason']. Increments the iteration counter and the run's
    llm_steps field.
    """
    settings = get_settings()
    run: AgentRun = state["run"]

    logger.info(
        "[llm_node] run=%s iteration=%d/%d",
        run.run_id, state["iterations"] + 1, settings.anthropic_max_iterations,
    )

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,  # type: ignore[arg-type]
        messages=state["messages"],
    )

    # Serialise the assistant message for the next turn
    content_blocks = []
    for block in response.content:
        if hasattr(block, "type"):
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

    state["messages"].append({"role": "assistant", "content": content_blocks})
    state["stop_reason"] = response.stop_reason
    state["iterations"] += 1
    run.llm_steps += 1

    logger.info(
        "[llm_node] stop_reason=%s content_blocks=%d",
        response.stop_reason, len(content_blocks),
    )

    return state


# ---------------------------------------------------------------------------
# Node: tool_node
# ---------------------------------------------------------------------------


async def tool_node(state: AgentState) -> AgentState:
    """
    Execute every tool_use block in the last assistant message in parallel.

    Special case: if apply_fix is called with human_approved=false, we do NOT
    execute it — instead we move the payload to state['pending_fix'] so the
    approval_node can persist it and pause the loop.

    Tool results are appended to state['messages'] as a user turn containing
    tool_result blocks (Anthropic's expected format).
    """
    run: AgentRun = state["run"]

    # Extract tool_use blocks from the last assistant message
    last_msg = state["messages"][-1]
    tool_use_blocks = [b for b in last_msg["content"] if b.get("type") == "tool_use"]

    if not tool_use_blocks:
        logger.warning("[tool_node] No tool_use blocks found — skipping.")
        return state

    import asyncio  # noqa: PLC0415

    async def _run_tool(block: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool and return a tool_result content block."""
        tid = block["id"]
        name = block["name"]
        inp = block["input"]

        # Record the call in the run trace
        tc = ToolCall(tool_use_id=tid, name=name, input=inp)
        run.add_tool_call(tc)

        # Special handling: apply_fix without approval → intercept
        if name == "apply_fix" and not inp.get("human_approved"):
            logger.info("[tool_node] apply_fix intercepted — requires approval. run=%s", run.run_id)
            state["pending_fix"] = inp
            result_content = json.dumps({
                "status": "pending_approval",
                "message": (
                    "Fix proposal recorded. The operator must approve this action via "
                    "POST /agent/approve before it can be executed."
                ),
            })
            is_error = False
        else:
            try:
                raw = await dispatch(name, inp)
                result_content = raw if isinstance(raw, str) else json.dumps(raw)
                is_error = False
            except Exception as exc:
                result_content = f"[{name}] ERROR: {exc}"
                is_error = True
                logger.warning("[tool_node] Tool %s raised: %s", name, exc)

        # Record the result in the run trace
        tr = ToolResult(
            tool_use_id=tid, name=name, output=result_content, is_error=is_error
        )
        run.add_tool_result(tr)

        return {
            "type": "tool_result",
            "tool_use_id": tid,
            "content": result_content,
            "is_error": is_error,
        }

    # Execute all tools concurrently
    results = await asyncio.gather(*[_run_tool(b) for b in tool_use_blocks])

    # Append as a single user turn with multiple tool_result blocks
    state["messages"].append({"role": "user", "content": list(results)})

    return state


# ---------------------------------------------------------------------------
# Node: approval_node
# ---------------------------------------------------------------------------


async def approval_node(state: AgentState) -> AgentState:
    """
    Persist the pending fix to the AgentRun and set status to AWAITING_APPROVAL.

    This is a terminal node for the current graph execution — the loop halts here
    and resumes only after the operator calls POST /agent/approve/{run_id}.
    """
    run: AgentRun = state["run"]
    run.pending_fix = state.get("pending_fix")
    run.set_status(AgentStatus.AWAITING_APPROVAL)

    logger.info(
        "[approval_node] run=%s paused awaiting approval for: %s",
        run.run_id,
        run.pending_fix,
    )
    return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def route_after_llm(
    state: AgentState,
) -> Literal["tool_node", "approval_node", "__end__"]:
    """
    Decide what comes after llm_node:

    - tool_use         → tool_node (run the requested tools)
    - end_turn         → END (Claude is done)
    - max_iterations   → END (safety cap)
    - pending_fix      → approval_node (intercept before next llm call)
    """
    settings = get_settings()

    if state["iterations"] >= settings.anthropic_max_iterations:
        logger.warning("[route] Max iterations (%d) reached — terminating.", settings.anthropic_max_iterations)
        return END

    if state.get("pending_fix"):
        return "approval_node"

    if state.get("stop_reason") == "tool_use":
        return "tool_node"

    return END


def route_after_tools(
    state: AgentState,
) -> Literal["llm_node", "approval_node", "__end__"]:
    """
    After tool results are appended:

    - pending_fix present → approval_node
    - otherwise           → llm_node (continue the loop)
    """
    if state.get("pending_fix"):
        return "approval_node"
    return "llm_node"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for the agent loop.

    Returns the compiled graph ready for async invocation.
    """
    graph = StateGraph(AgentState)

    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("approval_node", approval_node)

    graph.set_entry_point("llm_node")

    graph.add_conditional_edges(
        "llm_node",
        route_after_llm,
        {
            "tool_node": "tool_node",
            "approval_node": "approval_node",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "tool_node",
        route_after_tools,
        {
            "llm_node": "llm_node",
            "approval_node": "approval_node",
            END: END,
        },
    )

    graph.add_edge("approval_node", END)

    return graph.compile()


# Compiled graph singleton — built lazily on first import
_GRAPH = None


def get_graph():
    """Return the compiled LangGraph (built once, reused thereafter)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_graph(run: AgentRun) -> AgentRun:
    """
    Execute the agent loop for the given AgentRun.

    Called by the FastAPI background task in main.py. Initialises the graph
    state from the run, streams through nodes until termination, then
    extracts and persists the diagnosis and final status back to the run.

    Parameters
    ----------
    run : AgentRun
        The run record to execute. Must be in PENDING or RUNNING status.
        Modified in-place; the caller is responsible for persisting to Redis.

    Returns
    -------
    AgentRun — the updated run record.
    """
    initial_state: AgentState = AgentState(
        run=run,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Alert received: {run.alert_summary}\n"
                    + (f"Namespace: {run.namespace}\n" if run.namespace else "")
                    + (f"Cluster: {run.cluster}\n" if run.cluster else "")
                    + "\nPlease investigate and resolve this issue."
                ),
            }
        ],
        iterations=0,
        stop_reason=None,
        pending_fix=None,
    )

    graph = get_graph()

    logger.info("[run_graph] Starting agent loop for run=%s", run.run_id)
    final_state: AgentState = await graph.ainvoke(initial_state)

    # Extract the final assistant text as the diagnosis
    run = final_state["run"]
    for msg in reversed(final_state["messages"]):
        if msg.get("role") == "assistant":
            for block in msg.get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    run.diagnosis = block["text"]
                    break
            if run.diagnosis:
                break

    # Transition to COMPLETED only if not already in a terminal/approval state
    if run.status == AgentStatus.RUNNING:
        run.set_status(AgentStatus.COMPLETED)

    logger.info(
        "[run_graph] Finished run=%s status=%s iterations=%d",
        run.run_id, run.status, final_state["iterations"],
    )

    return run
