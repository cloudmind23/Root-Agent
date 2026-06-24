"""
Root Agent — FastAPI application entry point.

Exposes the HTTP API used by alert sources (PagerDuty, Prometheus Alertmanager, etc.)
and the operator dashboard to interact with the autonomous healing agent.

Endpoints
---------
POST /agent/analyze       — trigger a new analysis run
GET  /agent/runs/{run_id} — fetch the full state of a run
POST /agent/approve/{run_id} — approve or reject a pending cluster fix
GET  /agent/history       — list recent runs (paginated)
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, make_asgi_app

from app.models import (
    AgentRun,
    AgentRunSummary,
    AgentStatus,
    AnalyzeRequest,
    ApproveRequest,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root_agent.agent")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

RUNS_TOTAL = Counter(
    "root_agent_runs_total",
    "Total number of agent runs started",
    ["cluster", "status"],
)
RUN_DURATION = Histogram(
    "root_agent_run_duration_seconds",
    "Wall-clock duration of agent runs",
    ["cluster"],
)

# ---------------------------------------------------------------------------
# Application state (injected at startup)
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None
_REDIS_KEY_PREFIX = "root-agent:run:"
_HISTORY_INDEX = "root-agent:history"


def _run_key(run_id: str) -> str:
    """Return the Redis key for an AgentRun."""
    return f"{_REDIS_KEY_PREFIX}{run_id}"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to Redis on startup and close gracefully on shutdown."""
    global _redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    _redis = await aioredis.from_url(redis_url, decode_responses=True)
    logger.info("Connected to Redis at %s", redis_url)
    yield
    await _redis.aclose()
    logger.info("Redis connection closed.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    application = FastAPI(
        title="Root Agent API",
        description=(
            "Autonomous Kubernetes healing agent powered by Claude. "
            "Diagnoses cluster alerts and applies approved remediations."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Prometheus metrics at /metrics
    metrics_app = make_asgi_app()
    application.mount("/metrics", metrics_app)

    # Serve static dashboard
    static_dir = Path(__file__).parent / "static"
    application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return application


app = create_app()


# ---------------------------------------------------------------------------
# Helper: Redis persistence
# ---------------------------------------------------------------------------


async def _save_run(run: AgentRun) -> None:
    """Persist an AgentRun to Redis as JSON."""
    if _redis is None:
        raise RuntimeError("Redis not initialised — call inside a request context.")
    await _redis.set(_run_key(run.run_id), run.model_dump_json(), ex=86_400 * 7)  # TTL: 7 days
    # Prepend to the sorted history index (score = epoch timestamp)
    await _redis.zadd(
        _HISTORY_INDEX,
        {run.run_id: run.created_at.timestamp()},
    )


async def _load_run(run_id: str) -> AgentRun:
    """Load an AgentRun from Redis; raises 404 if not found."""
    if _redis is None:
        raise RuntimeError("Redis not initialised.")
    raw = await _redis.get(_run_key(run_id))
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return AgentRun.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Background task: agent loop
# ---------------------------------------------------------------------------


async def _run_agent_loop(run_id: str) -> None:
    """
    Execute the LangGraph-orchestrated agent loop for a given run.

    This coroutine is dispatched as a background task by POST /agent/analyze.
    It imports the graph lazily to avoid circular imports at module load time.

    The loop will:
      1. Build the initial state from the AgentRun stored in Redis.
      2. Stream LangGraph node executions (Claude inference → tool dispatch → repeat).
      3. Persist state updates after each node.
      4. Transition the run to COMPLETED, FAILED, or AWAITING_APPROVAL on exit.
    """
    try:
        # Lazy import — agent module is heavy (Anthropic client, K8s client)
        from app.agent.graph import run_graph  # noqa: PLC0415

        run = await _load_run(run_id)
        run.set_status(AgentStatus.RUNNING)
        await _save_run(run)

        start_ts = datetime.now(timezone.utc)
        updated_run = await run_graph(run)
        elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()

        await _save_run(updated_run)
        RUNS_TOTAL.labels(cluster=updated_run.cluster or "unknown", status=updated_run.status).inc()
        RUN_DURATION.labels(cluster=updated_run.cluster or "unknown").observe(elapsed)

    except Exception as exc:
        logger.exception("Agent loop failed for run %s: %s", run_id, exc)
        try:
            run = await _load_run(run_id)
            run.set_status(AgentStatus.FAILED)
            run.error_message = str(exc)
            await _save_run(run)
            RUNS_TOTAL.labels(cluster=run.cluster or "unknown", status=AgentStatus.FAILED).inc()
        except Exception:
            logger.exception("Could not persist FAILED status for run %s", run_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/agent/analyze",
    response_model=AgentRun,
    status_code=202,
    summary="Trigger a new agent analysis run",
    tags=["Agent"],
)
async def analyze(request: AnalyzeRequest) -> AgentRun:
    """
    Start a new autonomous diagnosis run.

    Creates an AgentRun record, persists it to Redis, fires the agent loop
    as a background task, and immediately returns the run with status=PENDING.

    The caller should poll GET /agent/runs/{run_id} to track progress.
    """
    from fastapi.background import BackgroundTasks  # noqa: PLC0415

    run_id = str(uuid.uuid4())
    run = AgentRun(
        run_id=run_id,
        alert_summary=request.alert_summary,
        namespace=request.namespace,
        cluster=request.cluster,
        status=AgentStatus.PENDING,
    )
    await _save_run(run)
    logger.info("Created run %s: %s", run_id, request.alert_summary[:80])

    # Kick off the agent loop without blocking the HTTP response
    import asyncio  # noqa: PLC0415
    asyncio.create_task(_run_agent_loop(run_id))

    RUNS_TOTAL.labels(cluster=run.cluster or "unknown", status=AgentStatus.PENDING).inc()
    return run


@app.get(
    "/agent/runs/{run_id}",
    response_model=AgentRun,
    summary="Get the current state of an agent run",
    tags=["Agent"],
)
async def get_run(run_id: str) -> AgentRun:
    """
    Fetch the full AgentRun record by ID.

    Returns the complete run including all tool call/result traces,
    the current status, diagnosis, and any pending fix payload.
    Raises 404 if the run ID does not exist.
    """
    return await _load_run(run_id)


@app.post(
    "/agent/approve/{run_id}",
    response_model=AgentRun,
    summary="Approve or reject a pending cluster fix",
    tags=["Agent"],
)
async def approve(run_id: str, request: ApproveRequest) -> AgentRun:
    """
    Human-in-the-loop gate for destructive cluster operations.

    When an agent run reaches status=AWAITING_APPROVAL, the operator
    must call this endpoint to either approve or reject the proposed fix.

    - approved=True  → resumes the agent loop to apply the patch/restart
    - approved=False → transitions the run to CANCELLED without touching the cluster

    Raises 409 if the run is not in AWAITING_APPROVAL state.
    """
    run = await _load_run(run_id)

    if run.status != AgentStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is not awaiting approval (current status: {run.status}).",
        )

    if not request.approved:
        logger.info(
            "Fix REJECTED for run %s by %s", run_id, request.approver or "unknown"
        )
        run.set_status(AgentStatus.CANCELLED)
        run.error_message = (
            f"Fix rejected by {request.approver or 'operator'} at {datetime.now(timezone.utc).isoformat()}"
        )
        await _save_run(run)
        return run

    logger.info(
        "Fix APPROVED for run %s by %s", run_id, request.approver or "unknown"
    )

    if run.pending_fix:
        # Inject the human_approved flag into the pending fix and resume
        run.pending_fix["human_approved"] = True
        run.pending_fix["_approved_by"] = request.approver
        await _save_run(run)
        import asyncio  # noqa: PLC0415
        asyncio.create_task(_run_agent_loop(run_id))

    return run


@app.get(
    "/agent/history",
    response_model=list[AgentRunSummary],
    summary="List recent agent runs",
    tags=["Agent"],
)
async def history(
    limit: int = Query(default=20, ge=1, le=100, description="Number of runs to return."),
    offset: int = Query(default=0, ge=0, description="Pagination offset."),
    status: AgentStatus | None = Query(default=None, description="Filter by run status."),
    cluster: str | None = Query(default=None, description="Filter by cluster name."),
) -> list[AgentRunSummary]:
    """
    Return a paginated list of recent agent runs, newest first.

    Supports optional filtering by status and cluster name.
    Each entry is a lightweight summary projection; use GET /agent/runs/{run_id}
    for full trace detail.
    """
    if _redis is None:
        raise HTTPException(status_code=503, detail="Storage not available.")

    # Fetch run IDs from the sorted set, newest first (descending score)
    run_ids: list[str] = await _redis.zrevrange(
        _HISTORY_INDEX, offset, offset + limit - 1
    )

    summaries: list[AgentRunSummary] = []
    for rid in run_ids:
        try:
            run = await _load_run(rid)
        except HTTPException:
            continue  # Stale index entry — skip

        if status and run.status != status:
            continue
        if cluster and run.cluster != cluster:
            continue

        summaries.append(
            AgentRunSummary(
                run_id=run.run_id,
                status=run.status,
                alert_summary=run.alert_summary,
                namespace=run.namespace,
                cluster=run.cluster,
                llm_steps=run.llm_steps,
                created_at=run.created_at,
                completed_at=run.completed_at,
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------


@app.get("/ui", tags=["Infra"], summary="Operator dashboard", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the single-page operator dashboard."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(str(static_dir / "dashboard.html"))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["Infra"], summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Return 200 OK when the service is running."""
    return {"status": "ok"}


@app.get("/readyz", tags=["Infra"], summary="Readiness probe")
async def readyz() -> dict[str, str]:
    """Return 200 OK only when Redis is reachable."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not connected.")
    try:
        await _redis.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis unreachable: {exc}") from exc
    return {"status": "ready"}
