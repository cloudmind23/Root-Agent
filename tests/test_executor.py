"""
Tests for app/tools/executor.py

Strategy
--------
All external dependencies (kubernetes client, httpx, subprocess) are mocked
at the point of import so no real cluster or network is needed.

Covers
------
- get_pod_logs   — happy path, tail truncation, previous flag, K8s error
- describe_resource — kubectl subprocess happy/fail, Python client fallback
- get_events     — filtered and unfiltered, empty result
- search_runbooks — happy path, HTTP error, connection error
- apply_fix      — approval gate (blocked), patch action, restart action, unknown action
- notify_slack   — sent, skipped (no webhook), HTTP error
- web_search     — happy path, no API key, HTTP error
- dispatch       — routes to correct handler, unknown tool name
- _truncate      — content within limit, content exceeding limit
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.executor import (
    _truncate,
    apply_fix,
    describe_resource,
    dispatch,
    get_events,
    get_pod_logs,
    notify_slack,
    search_runbooks,
    web_search,
)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_content_unchanged(self):
        text = "hello world"
        assert _truncate(text, max_chars=100) == text

    def test_long_content_truncated(self):
        text = "x" * 1000
        result = _truncate(text, max_chars=100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_truncated_preserves_head_and_tail(self):
        text = "A" * 500 + "B" * 500
        result = _truncate(text, max_chars=100)
        assert result.startswith("A")
        assert result.endswith("B")


# ---------------------------------------------------------------------------
# get_pod_logs
# ---------------------------------------------------------------------------


class TestGetPodLogs:
    @pytest.fixture
    def mock_k8s(self):
        """Patch get_k8s_clients to return a mock CoreV1Api."""
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            mock_settings.return_value.max_log_tail_lines = 500
            yield core_v1

    @pytest.mark.asyncio
    async def test_returns_logs(self, mock_k8s):
        mock_k8s.read_namespaced_pod_log.return_value = "INFO starting\nERROR oomkilled"
        result = await get_pod_logs("api-0", "prod")
        assert "INFO starting" in result
        assert "ERROR oomkilled" in result
        mock_k8s.read_namespaced_pod_log.assert_called_once_with(
            name="api-0",
            namespace="prod",
            container=None,
            tail_lines=100,
            previous=False,
            timestamps=True,
        )

    @pytest.mark.asyncio
    async def test_previous_flag_passed(self, mock_k8s):
        mock_k8s.read_namespaced_pod_log.return_value = "crash logs"
        await get_pod_logs("api-0", "prod", previous=True)
        _, kwargs = mock_k8s.read_namespaced_pod_log.call_args
        assert kwargs["previous"] is True

    @pytest.mark.asyncio
    async def test_container_name_passed(self, mock_k8s):
        mock_k8s.read_namespaced_pod_log.return_value = "logs"
        await get_pod_logs("api-0", "prod", container="sidecar")
        _, kwargs = mock_k8s.read_namespaced_pod_log.call_args
        assert kwargs["container"] == "sidecar"

    @pytest.mark.asyncio
    async def test_tail_lines_capped_by_settings(self, mock_k8s):
        mock_k8s.read_namespaced_pod_log.return_value = "logs"
        # Request 9999 lines but settings cap at 500
        await get_pod_logs("api-0", "prod", tail_lines=9999)
        _, kwargs = mock_k8s.read_namespaced_pod_log.call_args
        assert kwargs["tail_lines"] == 500

    @pytest.mark.asyncio
    async def test_k8s_error_returns_error_string(self, mock_k8s):
        mock_k8s.read_namespaced_pod_log.side_effect = Exception("pod not found")
        result = await get_pod_logs("missing-pod", "prod")
        assert "ERROR" in result
        assert "missing-pod" in result

    @pytest.mark.asyncio
    async def test_empty_logs_returns_placeholder(self, mock_k8s):
        mock_k8s.read_namespaced_pod_log.return_value = ""
        result = await get_pod_logs("api-0", "prod")
        assert "no logs" in result.lower()


# ---------------------------------------------------------------------------
# describe_resource
# ---------------------------------------------------------------------------


class TestDescribeResource:
    @pytest.mark.asyncio
    async def test_kubectl_happy_path(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Name: api\nNamespace: prod\n"
        with patch("subprocess.run", return_value=mock_result):
            result = await describe_resource("deployment", "api", "prod")
        assert "Name: api" in result

    @pytest.mark.asyncio
    async def test_kubectl_nonzero_returns_stderr(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: deployment not found"
        with patch("subprocess.run", return_value=mock_result):
            result = await describe_resource("deployment", "missing", "prod")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_kubectl_not_found_falls_back_to_python_client(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            apps_v1.read_namespaced_deployment.return_value = MagicMock(__str__=lambda s: "deployment_obj")
            with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
                result = await describe_resource("deployment", "api", "prod")
        assert result  # non-empty

    @pytest.mark.asyncio
    async def test_cluster_scoped_resource_omits_namespace_flag(self):
        mock_result = MagicMock(returncode=0, stdout="node info", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await describe_resource("node", "worker-1", "all")
        args = mock_run.call_args[0][0]
        assert "-n" not in args


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


class TestGetEvents:
    def _make_event(self, namespace="prod", reason="OOMKilling", kind="Pod", name="api-0", message="OOM"):
        ev = MagicMock()
        ev.metadata.namespace = namespace
        ev.reason = reason
        ev.involved_object.kind = kind
        ev.involved_object.name = name
        ev.message = message
        return ev

    @pytest.mark.asyncio
    async def test_returns_formatted_table(self):
        core_v1 = MagicMock()
        core_v1.list_namespaced_event.return_value.items = [
            self._make_event()
        ]
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, MagicMock())):
            result = await get_events(namespace="prod")
        assert "OOMKilling" in result
        assert "Pod/api-0" in result

    @pytest.mark.asyncio
    async def test_empty_events_returns_message(self):
        core_v1 = MagicMock()
        core_v1.list_namespaced_event.return_value.items = []
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, MagicMock())):
            result = await get_events(namespace="prod")
        assert "No events" in result

    @pytest.mark.asyncio
    async def test_all_namespaces_calls_list_for_all(self):
        core_v1 = MagicMock()
        core_v1.list_event_for_all_namespaces.return_value.items = []
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, MagicMock())):
            await get_events(namespace="all")
        core_v1.list_event_for_all_namespaces.assert_called_once()
        core_v1.list_namespaced_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_resource_name_filter_in_field_selector(self):
        core_v1 = MagicMock()
        core_v1.list_namespaced_event.return_value.items = []
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, MagicMock())):
            await get_events(namespace="prod", resource_name="api-0")
        _, kwargs = core_v1.list_namespaced_event.call_args
        assert "api-0" in kwargs.get("field_selector", "")

    @pytest.mark.asyncio
    async def test_k8s_error_returns_error_string(self):
        core_v1 = MagicMock()
        core_v1.list_namespaced_event.side_effect = Exception("forbidden")
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, MagicMock())):
            result = await get_events(namespace="prod")
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# search_runbooks
# ---------------------------------------------------------------------------


class TestSearchRunbooks:
    @pytest.mark.asyncio
    async def test_happy_path_returns_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"title": "OOM Runbook", "content": "..."}]}
        mock_response.raise_for_status = MagicMock()

        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.rag_api_url = "http://rag-api:8000"
            mock_settings.return_value.rag_api_timeout = 10.0
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await search_runbooks("OOMKilled production")

        assert "results" in result
        assert result["results"][0]["title"] == "OOM Runbook"

    @pytest.mark.asyncio
    async def test_http_error_returns_error_dict(self):
        import httpx as _httpx

        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.rag_api_url = "http://rag-api:8000"
            mock_settings.return_value.rag_api_timeout = 10.0
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_resp.text = "Service Unavailable"
            mock_client.post = AsyncMock(
                side_effect=_httpx.HTTPStatusError("503", request=MagicMock(), response=mock_resp)
            )
            mock_client_cls.return_value = mock_client

            result = await search_runbooks("something")

        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_connection_error_returns_error_dict(self):
        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.rag_api_url = "http://rag-api:8000"
            mock_settings.return_value.rag_api_timeout = 10.0
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_client_cls.return_value = mock_client

            result = await search_runbooks("something")

        assert "error" in result


# ---------------------------------------------------------------------------
# apply_fix
# ---------------------------------------------------------------------------


class TestApplyFix:
    @pytest.mark.asyncio
    async def test_blocked_when_not_approved(self):
        result = await apply_fix(
            action="patch",
            resource_type="deployment",
            resource_name="api",
            human_approved=False,
        )
        assert result["status"] == "blocked"
        assert "approval" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_patch_deployment(self):
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            result = await apply_fix(
                action="patch",
                resource_type="deployment",
                resource_name="api",
                human_approved=True,
                namespace="prod",
                patch_body={"spec": {"replicas": 3}},
            )
        assert result["status"] == "success"
        assert result["action"] == "patch"
        apps_v1.patch_namespaced_deployment.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_deployment(self):
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            result = await apply_fix(
                action="restart",
                resource_type="deployment",
                resource_name="api",
                human_approved=True,
                namespace="prod",
            )
        assert result["status"] == "success"
        assert result["action"] == "restart"
        apps_v1.patch_namespaced_deployment.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_pod_deletes_pod(self):
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            result = await apply_fix(
                action="restart",
                resource_type="pod",
                resource_name="api-0",
                human_approved=True,
                namespace="prod",
            )
        assert result["status"] == "success"
        assert result["action"] == "deleted_pod"
        core_v1.delete_namespaced_pod.assert_called_once_with("api-0", "prod")

    @pytest.mark.asyncio
    async def test_patch_missing_body_returns_error(self):
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            result = await apply_fix(
                action="patch",
                resource_type="deployment",
                resource_name="api",
                human_approved=True,
            )
        assert result["status"] == "error"
        assert "patch_body" in result["message"]

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            result = await apply_fix(
                action="delete",  # not a supported action
                resource_type="deployment",
                resource_name="api",
                human_approved=True,
            )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_k8s_exception_returns_error(self):
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        apps_v1.patch_namespaced_deployment.side_effect = Exception("forbidden")
        with patch("app.core.config.get_k8s_clients", return_value=(core_v1, apps_v1)):
            result = await apply_fix(
                action="patch",
                resource_type="deployment",
                resource_name="api",
                human_approved=True,
                patch_body={"spec": {}},
            )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# notify_slack
# ---------------------------------------------------------------------------


class TestNotifySlack:
    @pytest.mark.asyncio
    async def test_skipped_when_no_webhook(self):
        with patch("app.tools.executor.get_settings") as mock_settings:
            mock_settings.return_value.slack_webhook_url = None
            result = await notify_slack("alert", severity="warning")
        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_sends_message(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.slack_webhook_url = "https://hooks.slack.com/T123"
            mock_settings.return_value.slack_timeout = 5.0
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await notify_slack("Pod OOMKilled", severity="critical", run_id="run-1")

        assert result["status"] == "sent"
        assert result["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_http_error_returns_error(self):
        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.slack_webhook_url = "https://hooks.slack.com/T123"
            mock_settings.return_value.slack_timeout = 5.0
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("timeout"))
            mock_client_cls.return_value = mock_client

            result = await notify_slack("msg", severity="info")

        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_unavailable_when_no_api_key(self):
        with patch("app.tools.executor.get_settings") as mock_settings:
            mock_settings.return_value.web_search_api_key = None
            result = await web_search("OOMKilled kubernetes")
        assert result["status"] == "unavailable"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_returns_organic_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {"title": "K8s OOM docs", "link": "https://k8s.io/oom", "snippet": "Memory limits..."},
                {"title": "GH Issue #123", "link": "https://github.com/...", "snippet": "Fix: increase limit"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.web_search_api_key = "key123"
            mock_settings.return_value.web_search_url = "https://google.serper.dev/search"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await web_search("OOMKilled kubernetes")

        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "K8s OOM docs"

    @pytest.mark.asyncio
    async def test_caps_num_results_at_10(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status = MagicMock()

        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.web_search_api_key = "key123"
            mock_settings.return_value.web_search_url = "https://google.serper.dev/search"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await web_search("query", num_results=99)
            _, kwargs = mock_client.post.call_args
            assert kwargs["json"]["num"] == 10

    @pytest.mark.asyncio
    async def test_http_error_returns_error(self):
        with patch("app.tools.executor.get_settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.return_value.web_search_api_key = "key123"
            mock_settings.return_value.web_search_url = "https://google.serper.dev/search"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("network error"))
            mock_client_cls.return_value = mock_client

            result = await web_search("query")

        assert result["status"] == "error"
        assert result["results"] == []


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    @pytest.mark.asyncio
    async def test_routes_to_correct_handler(self):
        mock_handler = AsyncMock(return_value="logs")
        with patch.dict("app.tools.executor.TOOL_HANDLERS", {"get_pod_logs": mock_handler}):
            result = await dispatch("get_pod_logs", {"pod_name": "api-0", "namespace": "prod"})
        mock_handler.assert_awaited_once_with(pod_name="api-0", namespace="prod")
        assert result == "logs"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_dict(self):
        result = await dispatch("nonexistent_tool", {})
        assert isinstance(result, dict)
        assert "error" in result
        assert "nonexistent_tool" in result["error"]
