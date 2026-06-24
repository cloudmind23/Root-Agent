"""
Claude API tool definitions for the Root Agent autonomous Kubernetes healing agent.

Each tool is defined as a dict conforming to the Anthropic tool-use JSON schema.
The LLM uses the description and input_schema to decide when and how to call each tool.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

GET_POD_LOGS: dict[str, Any] = {
    "name": "get_pod_logs",
    "description": (
        "Fetch stdout/stderr logs from a specific Kubernetes pod. "
        "Use this tool first when investigating a pod that is CrashLooping, "
        "OOMKilled, or reporting application-level errors. "
        "Supports tail-line limiting and optional container selection for multi-container pods."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pod_name": {
                "type": "string",
                "description": "Name of the pod to fetch logs from.",
            },
            "namespace": {
                "type": "string",
                "description": "Kubernetes namespace the pod lives in (e.g. 'default', 'production').",
            },
            "container": {
                "type": "string",
                "description": (
                    "Container name within the pod. Required only for multi-container pods. "
                    "Omit for single-container pods."
                ),
            },
            "tail_lines": {
                "type": "integer",
                "description": "Number of log lines to return from the tail. Defaults to 100.",
                "default": 100,
            },
            "previous": {
                "type": "boolean",
                "description": (
                    "If true, fetch logs from the previous (crashed) container instance. "
                    "Useful for diagnosing CrashLoopBackOff."
                ),
                "default": False,
            },
        },
        "required": ["pod_name", "namespace"],
    },
}

DESCRIBE_RESOURCE: dict[str, Any] = {
    "name": "describe_resource",
    "description": (
        "Run 'kubectl describe' on any Kubernetes resource to get its full spec, "
        "status, conditions, and recent events. "
        "Use this to inspect Deployments, Pods, Services, Nodes, PersistentVolumeClaims, "
        "or any other resource kind when you need detailed state information beyond what "
        "get_pod_logs provides. Particularly useful for diagnosing scheduling failures, "
        "readiness/liveness probe failures, and resource quota issues."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resource_type": {
                "type": "string",
                "description": (
                    "Kubernetes resource kind, e.g. 'pod', 'deployment', 'service', "
                    "'node', 'persistentvolumeclaim', 'replicaset'."
                ),
            },
            "resource_name": {
                "type": "string",
                "description": "Name of the specific resource to describe.",
            },
            "namespace": {
                "type": "string",
                "description": (
                    "Namespace of the resource. Use 'all' for cluster-scoped resources "
                    "like nodes. Defaults to 'default'."
                ),
                "default": "default",
            },
        },
        "required": ["resource_type", "resource_name"],
    },
}

GET_EVENTS: dict[str, Any] = {
    "name": "get_events",
    "description": (
        "Pull Kubernetes cluster events, optionally filtered by namespace or a specific "
        "resource name. Events capture warnings and state changes such as "
        "OOMKilled, BackOff, FailedScheduling, and Unhealthy probe results. "
        "Use this when you need a timeline of what happened to a resource or namespace, "
        "especially when pod logs are insufficient or the container never started."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": (
                    "Namespace to filter events for. "
                    "Pass 'all' to retrieve cluster-wide events."
                ),
                "default": "default",
            },
            "resource_name": {
                "type": "string",
                "description": (
                    "Optional. Filter events related to a specific resource by name "
                    "(e.g. a pod or deployment name)."
                ),
            },
            "event_type": {
                "type": "string",
                "enum": ["Warning", "Normal", "all"],
                "description": "Filter by event type. Use 'Warning' to focus on problems.",
                "default": "Warning",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of events to return. Defaults to 50.",
                "default": 50,
            },
        },
        "required": [],
    },
}

SEARCH_RUNBOOKS: dict[str, Any] = {
    "name": "search_runbooks",
    "description": (
        "Query the internal RAG (retrieval-augmented generation) knowledge base to find "
        "runbooks, playbooks, and documented remediation steps relevant to a Kubernetes "
        "error or symptom. "
        "Use this tool after identifying an error message or failure mode to check whether "
        "a known fix or escalation procedure already exists before attempting autonomous remediation. "
        "Returns ranked runbook excerpts with source metadata."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the error or symptom to search for. "
                    "Include the error message, affected resource type, and any relevant context. "
                    "Example: 'CrashLoopBackOff due to OOMKilled in namespace production'"
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of runbook results to return. Defaults to 3.",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}

APPLY_FIX: dict[str, Any] = {
    "name": "apply_fix",
    "description": (
        "Apply a remediation action to a Kubernetes cluster — either patch a resource spec "
        "or restart a pod/deployment. "
        "IMPORTANT: This tool makes live changes to the cluster. It MUST only be called "
        "when human_approved is true. Never set human_approved=true yourself; wait for the "
        "operator to approve the action via the /agent/approve endpoint before invoking this tool. "
        "Supported actions: 'patch' to apply a strategic-merge or JSON patch, "
        "'restart' to perform a rollout restart of a deployment or statefulset."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["patch", "restart"],
                "description": (
                    "'patch' applies a JSON/strategic-merge patch to a resource. "
                    "'restart' triggers a rollout restart (sets restartedAt annotation)."
                ),
            },
            "resource_type": {
                "type": "string",
                "description": "Kubernetes resource kind to target, e.g. 'deployment', 'statefulset', 'pod'.",
            },
            "resource_name": {
                "type": "string",
                "description": "Name of the resource to modify.",
            },
            "namespace": {
                "type": "string",
                "description": "Namespace of the target resource.",
                "default": "default",
            },
            "patch_body": {
                "type": "object",
                "description": (
                    "The patch payload (required when action='patch'). "
                    "Must be a valid strategic-merge or JSON patch dict. "
                    "Example to increase memory limit: "
                    "{\"spec\": {\"template\": {\"spec\": {\"containers\": [{\"name\": \"app\", "
                    "\"resources\": {\"limits\": {\"memory\": \"512Mi\"}}}]}}}}"
                ),
            },
            "human_approved": {
                "type": "boolean",
                "description": (
                    "Must be true before any cluster mutation is executed. "
                    "Do NOT set this to true autonomously — it must be set by the "
                    "human operator via the approval endpoint."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Human-readable explanation of why this fix is being applied. Logged for audit.",
            },
        },
        "required": ["action", "resource_type", "resource_name", "human_approved"],
    },
}

NOTIFY_SLACK: dict[str, Any] = {
    "name": "notify_slack",
    "description": (
        "Send an alert or status message to the configured Slack webhook. "
        "Use this tool to notify the on-call team when: "
        "(1) a critical issue is detected that requires human intervention, "
        "(2) an autonomous fix has been applied, "
        "(3) the agent needs human approval before proceeding, or "
        "(4) the investigation is complete and a summary should be shared. "
        "Keep messages concise and action-oriented."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "The message body to send to Slack. "
                    "Supports Slack mrkdwn formatting (*bold*, _italic_, `code`, ```blocks```)."
                ),
            },
            "severity": {
                "type": "string",
                "enum": ["info", "warning", "critical"],
                "description": (
                    "Severity level that controls the message color attachment: "
                    "info=blue, warning=yellow, critical=red."
                ),
                "default": "info",
            },
            "run_id": {
                "type": "string",
                "description": "Agent run ID to include in the notification for traceability.",
            },
            "cluster": {
                "type": "string",
                "description": "Name of the Kubernetes cluster this alert pertains to.",
            },
        },
        "required": ["message", "severity"],
    },
}

WEB_SEARCH: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "Search the web for documentation, GitHub issues, or community solutions related to "
        "an unknown or novel Kubernetes error. "
        "Use this as a fallback when get_pod_logs and search_runbooks have not produced a "
        "clear remediation path — for example, when encountering an obscure error code, "
        "a third-party controller error, or a Kubernetes version-specific regression. "
        "Prefer official Kubernetes docs, GitHub issues, and Stack Overflow results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query string. Include the exact error message in quotes when possible, "
                    "along with relevant context such as Kubernetes version or resource type. "
                    "Example: '\"Back-off restarting failed container\" kubernetes OOMKilled init container'"
                ),
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to return. Defaults to 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Master list — pass this to the Anthropic client as the `tools` parameter
# ---------------------------------------------------------------------------

ALL_TOOLS: list[dict[str, Any]] = [
    GET_POD_LOGS,
    DESCRIBE_RESOURCE,
    GET_EVENTS,
    SEARCH_RUNBOOKS,
    APPLY_FIX,
    NOTIFY_SLACK,
    WEB_SEARCH,
]
