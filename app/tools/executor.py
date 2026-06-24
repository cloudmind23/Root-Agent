"""
Tool executor — bridges Claude's tool_use requests to real infrastructure calls.

Each public function maps 1-to-1 with a tool defined in app/tools/definitions.py.
The dispatch table at the bottom routes tool names to their implementations.

All functions are async and return a plain string or dict that is safe to
serialise back to the Anthropic API as a tool_result content block.
Error handling is intentionally wide-catch so that a tool failure surfaces as
an informational error string rather than crashing the agent loop.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger("root_agent.executor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int = 40_000) -> str:
    """Truncate output to stay within Claude's context window."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n" + text[-half:]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# get_pod_logs
# ---------------------------------------------------------------------------


async def get_pod_logs(
    pod_name: str,
    namespace: str,
    container: str | None = None,
    tail_lines: int = 100,
    previous: bool = False,
) -> str:
    """
    Fetch stdout/stderr from a Kubernetes pod via the Python K8s client.

    Parameters
    ----------
    pod_name:   Name of the target pod.
    namespace:  Namespace the pod lives in.
    container:  Container name (required for multi-container pods).
    tail_lines: Number of lines to return from the tail.
    previous:   Fetch logs from the previous (crashed) container instance.

    Returns
    -------
    Raw log text or an error string.
    """
    settings = get_settings()
    tail_lines = min(tail_lines, settings.max_log_tail_lines)

    try:
        from app.core.config import get_k8s_clients

        core_v1, _ = get_k8s_clients()
        logs: str = core_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container,
            tail_lines=tail_lines,
            previous=previous,
            timestamps=True,
        )
        return _truncate(logs or "(no logs returned)")
    except Exception as exc:
        error = f"[get_pod_logs ERROR] pod={pod_name} ns={namespace}: {exc}"
        logger.warning(error)
        return error


# ---------------------------------------------------------------------------
# describe_resource
# ---------------------------------------------------------------------------


async def describe_resource(
    resource_type: str,
    resource_name: str,
    namespace: str = "default",
) -> str:
    """
    Run the equivalent of `kubectl describe <resource_type> <resource_name>` by
    calling the K8s API and formatting the result as human-readable YAML-ish text.

    Falls back to shelling out to kubectl if the Python client doesn't have a
    dedicated method for the resource kind.

    Parameters
    ----------
    resource_type:  K8s resource kind (pod, deployment, service, node, …).
    resource_name:  Name of the specific resource.
    namespace:      Namespace, or 'all' for cluster-scoped resources.
    """
    import subprocess  # noqa: PLC0415

    ns_flag = [] if namespace.lower() == "all" else ["-n", namespace]

    try:
        result = subprocess.run(
            ["kubectl", "describe", resource_type, resource_name] + ns_flag,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return _truncate(output or "(no output)")
    except FileNotFoundError:
        # kubectl not on PATH — fall back to Python client for common types
        return await _describe_via_python_client(resource_type, resource_name, namespace)
    except Exception as exc:
        error = f"[describe_resource ERROR] {resource_type}/{resource_name}: {exc}"
        logger.warning(error)
        return error


async def _describe_via_python_client(
    resource_type: str, resource_name: str, namespace: str
) -> str:
    """
    Fallback: use the Python K8s client for common resource kinds.
    Returns JSON-serialised resource dict.
    """
    try:
        from app.core.config import get_k8s_clients  # noqa: PLC0415

        core_v1, apps_v1 = get_k8s_clients()
        rtype = resource_type.lower()

        if rtype == "pod":
            obj = core_v1.read_namespaced_pod(resource_name, namespace)
        elif rtype == "deployment":
            obj = apps_v1.read_namespaced_deployment(resource_name, namespace)
        elif rtype == "service":
            obj = core_v1.read_namespaced_service(resource_name, namespace)
        elif rtype == "node":
            obj = core_v1.read_node(resource_name)
        elif rtype in ("persistentvolumeclaim", "pvc"):
            obj = core_v1.read_namespaced_persistent_volume_claim(resource_name, namespace)
        else:
            return f"[describe_resource] Unsupported resource type via Python client: {resource_type}"

        return _truncate(str(obj))
    except Exception as exc:
        return f"[describe_resource ERROR via Python client] {resource_type}/{resource_name}: {exc}"


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


async def get_events(
    namespace: str = "default",
    resource_name: str | None = None,
    event_type: str = "Warning",
    limit: int = 50,
) -> str:
    """
    Pull Kubernetes events, optionally filtered by namespace, resource name, and type.

    Parameters
    ----------
    namespace:      Namespace to list events from ('all' for cluster-wide).
    resource_name:  If set, only return events involving this resource.
    event_type:     'Warning', 'Normal', or 'all'.
    limit:          Maximum number of events to return.
    """
    try:
        from app.core.config import get_k8s_clients  # noqa: PLC0415

        core_v1, _ = get_k8s_clients()

        kwargs: dict[str, Any] = {"limit": limit}
        if event_type.lower() != "all":
            kwargs["field_selector"] = f"type={event_type}"
        if resource_name:
            existing = kwargs.get("field_selector", "")
            sep = "," if existing else ""
            kwargs["field_selector"] = existing + sep + f"involvedObject.name={resource_name}"

        if namespace.lower() == "all":
            event_list = core_v1.list_event_for_all_namespaces(**kwargs)
        else:
            event_list = core_v1.list_namespaced_event(namespace, **kwargs)

        if not event_list.items:
            return "No events found matching the specified filters."

        lines: list[str] = [
            f"{'NAMESPACE':<20} {'REASON':<25} {'OBJECT':<35} {'MESSAGE'}"
        ]
        lines.append("-" * 120)
        for ev in event_list.items:
            ns = ev.metadata.namespace or ""
            reason = ev.reason or ""
            obj = f"{ev.involved_object.kind}/{ev.involved_object.name}"
            msg = (ev.message or "").replace("\n", " ")
            lines.append(f"{ns:<20} {reason:<25} {obj:<35} {msg}")

        return _truncate("\n".join(lines))
    except Exception as exc:
        error = f"[get_events ERROR] ns={namespace}: {exc}"
        logger.warning(error)
        return error


# ---------------------------------------------------------------------------
# search_runbooks
# ---------------------------------------------------------------------------


async def search_runbooks(query: str, top_k: int = 3) -> dict[str, Any]:
    """
    Query the internal RAG API for runbooks matching the given error description.

    Parameters
    ----------
    query:  Natural-language description of the error or symptom.
    top_k:  Number of results to return.

    Returns
    -------
    dict with keys 'results' (list of runbook hits) or 'error'.
    """
    settings = get_settings()
    url = f"{settings.rag_api_url}/query"

    try:
        async with httpx.AsyncClient(timeout=settings.rag_api_timeout) as client:
            response = await client.post(
                url,
                json={"query": query, "top_k": top_k},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        error = f"RAG API returned {exc.response.status_code}: {exc.response.text[:200]}"
        logger.warning("[search_runbooks] %s", error)
        return {"error": error, "results": []}
    except Exception as exc:
        error = f"RAG API unreachable: {exc}"
        logger.warning("[search_runbooks] %s", error)
        return {"error": error, "results": []}


# ---------------------------------------------------------------------------
# apply_fix
# ---------------------------------------------------------------------------


async def apply_fix(
    action: str,
    resource_type: str,
    resource_name: str,
    human_approved: bool,
    namespace: str = "default",
    patch_body: dict[str, Any] | None = None,
    reason: str | None = None,
    **_: Any,  # absorb internal keys like _approved_by
) -> dict[str, Any]:
    """
    Apply a remediation action to the cluster.

    SAFETY: This function refuses to execute if human_approved is not True.
    The caller (agent loop) is responsible for ensuring approval was granted
    via the /agent/approve endpoint before this tool is invoked.

    Parameters
    ----------
    action:         'patch' or 'restart'.
    resource_type:  K8s resource kind (deployment, pod, statefulset, …).
    resource_name:  Name of the resource to modify.
    human_approved: Must be True; checked at runtime as a safety fence.
    namespace:      Namespace of the resource.
    patch_body:     Strategic-merge patch dict (required for action='patch').
    reason:         Audit log reason string.
    """
    if not human_approved:
        return {
            "status": "blocked",
            "message": (
                "This action requires human approval. "
                "Please request approval via the /agent/approve endpoint before retrying."
            ),
        }

    logger.info(
        "Applying fix: action=%s resource=%s/%s ns=%s reason=%s approved_at=%s",
        action, resource_type, resource_name, namespace, reason, _utcnow_iso(),
    )

    try:
        from app.core.config import get_k8s_clients  # noqa: PLC0415

        core_v1, apps_v1 = get_k8s_clients()
        rtype = resource_type.lower()

        if action == "restart":
            return await _restart_resource(apps_v1, core_v1, rtype, resource_name, namespace)
        elif action == "patch":
            if not patch_body:
                return {"status": "error", "message": "patch_body is required for action='patch'"}
            return await _patch_resource(apps_v1, core_v1, rtype, resource_name, namespace, patch_body)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    except Exception as exc:
        logger.exception("[apply_fix ERROR]")
        return {"status": "error", "message": str(exc)}


async def _restart_resource(
    apps_v1: Any, core_v1: Any, rtype: str, name: str, namespace: str
) -> dict[str, Any]:
    """Perform a rollout restart by patching the restartedAt annotation."""
    restart_patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": _utcnow_iso()
                    }
                }
            }
        }
    }

    if rtype == "deployment":
        apps_v1.patch_namespaced_deployment(name, namespace, restart_patch)
    elif rtype == "statefulset":
        apps_v1.patch_namespaced_stateful_set(name, namespace, restart_patch)
    elif rtype == "daemonset":
        apps_v1.patch_namespaced_daemon_set(name, namespace, restart_patch)
    elif rtype == "pod":
        # Pods can't be restarted in-place — delete and let the controller recreate
        core_v1.delete_namespaced_pod(name, namespace)
        return {"status": "success", "action": "deleted_pod", "resource": f"{namespace}/{name}"}
    else:
        return {"status": "error", "message": f"Restart not supported for resource type: {rtype}"}

    return {"status": "success", "action": "restart", "resource": f"{namespace}/{rtype}/{name}"}


async def _patch_resource(
    apps_v1: Any, core_v1: Any, rtype: str, name: str, namespace: str, patch_body: dict[str, Any]
) -> dict[str, Any]:
    """Apply a strategic-merge patch to a resource."""
    if rtype == "deployment":
        apps_v1.patch_namespaced_deployment(name, namespace, patch_body)
    elif rtype == "statefulset":
        apps_v1.patch_namespaced_stateful_set(name, namespace, patch_body)
    elif rtype == "service":
        core_v1.patch_namespaced_service(name, namespace, patch_body)
    elif rtype == "pod":
        core_v1.patch_namespaced_pod(name, namespace, patch_body)
    elif rtype in ("persistentvolumeclaim", "pvc"):
        core_v1.patch_namespaced_persistent_volume_claim(name, namespace, patch_body)
    else:
        return {"status": "error", "message": f"Patch not supported for resource type: {rtype}"}

    return {
        "status": "success",
        "action": "patch",
        "resource": f"{namespace}/{rtype}/{name}",
        "patch_applied": patch_body,
    }


# ---------------------------------------------------------------------------
# notify_slack
# ---------------------------------------------------------------------------


async def notify_slack(
    message: str,
    severity: str = "info",
    run_id: str | None = None,
    cluster: str | None = None,
) -> dict[str, Any]:
    """
    Send a message to the configured Slack incoming webhook.

    Parameters
    ----------
    message:   Slack mrkdwn-formatted message body.
    severity:  'info' | 'warning' | 'critical' — controls attachment colour.
    run_id:    Agent run ID for traceability.
    cluster:   Cluster name for context.
    """
    settings = get_settings()

    if not settings.slack_webhook_url:
        msg = "SLACK_WEBHOOK_URL not configured — notification skipped."
        logger.warning("[notify_slack] %s", msg)
        return {"status": "skipped", "reason": msg}

    color_map = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
    color = color_map.get(severity, "#36a64f")

    footer_parts = []
    if run_id:
        footer_parts.append(f"run: {run_id}")
    if cluster:
        footer_parts.append(f"cluster: {cluster}")

    payload: dict[str, Any] = {
        "attachments": [
            {
                "color": color,
                "text": message,
                "footer": " | ".join(footer_parts) if footer_parts else "Root Agent",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=settings.slack_timeout) as client:
            response = await client.post(settings.slack_webhook_url, json=payload)
            response.raise_for_status()
            return {"status": "sent", "severity": severity}
    except Exception as exc:
        error = f"Slack delivery failed: {exc}"
        logger.warning("[notify_slack] %s", error)
        return {"status": "error", "message": error}


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


async def web_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """
    Search the web via the Serper.dev Google Search API.

    Falls back to a stub response if the API key is not configured so the
    agent loop can continue degraded rather than crashing.

    Parameters
    ----------
    query:       Search query string. Include exact error messages in quotes.
    num_results: Number of result items to return (max 10).
    """
    settings = get_settings()

    if not settings.web_search_api_key:
        logger.warning("[web_search] SERPER_API_KEY not configured — returning stub.")
        return {
            "status": "unavailable",
            "message": "Web search is not configured (missing SERPER_API_KEY).",
            "results": [],
        }

    num_results = min(num_results, 10)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                settings.web_search_url,
                headers={
                    "X-API-KEY": settings.web_search_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num_results},
            )
            response.raise_for_status()
            data = response.json()

        organic = data.get("organic", [])
        results = [
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in organic[:num_results]
        ]
        return {"status": "ok", "query": query, "results": results}

    except Exception as exc:
        error = f"Web search failed: {exc}"
        logger.warning("[web_search] %s", error)
        return {"status": "error", "message": error, "results": []}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

TOOL_HANDLERS: dict[str, Any] = {
    "get_pod_logs": get_pod_logs,
    "describe_resource": describe_resource,
    "get_events": get_events,
    "search_runbooks": search_runbooks,
    "apply_fix": apply_fix,
    "notify_slack": notify_slack,
    "web_search": web_search,
}


async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> str | dict[str, Any]:
    """
    Route a tool_use request from Claude to the correct implementation.

    Parameters
    ----------
    tool_name:  Must be one of the keys in TOOL_HANDLERS.
    tool_input: Parsed arguments from the Claude tool_use block.

    Returns
    -------
    String or dict — safe to embed in an Anthropic tool_result content block.

    Raises
    ------
    KeyError if tool_name is not registered.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {
            "error": f"Unknown tool '{tool_name}'. "
            f"Available tools: {list(TOOL_HANDLERS.keys())}"
        }
    return await handler(**tool_input)
