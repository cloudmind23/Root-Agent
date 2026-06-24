"""
Centralised configuration and shared client factories for Root Agent.

Settings are loaded from environment variables (with .env file support).
Kubernetes and HTTP clients are constructed lazily and cached as module-level
singletons so callers don't need to manage client lifecycle.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger("root_agent.config")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings:
    """
    Application settings resolved from environment variables.

    All values have sensible defaults for local development.
    Production deployments should provide real values via environment or
    a Kubernetes Secret mounted as env vars.
    """

    def __init__(self) -> None:
        self.anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.anthropic_max_tokens: int = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))
        self.anthropic_max_iterations: int = int(os.environ.get("ANTHROPIC_MAX_ITERATIONS", "15"))

        self.redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379")

        self.rag_api_url: str = os.environ.get("RAG_API_URL", "http://rag-api:8000")
        self.rag_api_timeout: float = float(os.environ.get("RAG_API_TIMEOUT", "10.0"))

        self.slack_webhook_url: Optional[str] = os.environ.get("SLACK_WEBHOOK_URL")
        self.slack_timeout: float = float(os.environ.get("SLACK_TIMEOUT", "5.0"))

        self.web_search_api_key: Optional[str] = os.environ.get("SERPER_API_KEY")
        self.web_search_url: str = "https://google.serper.dev/search"

        # Kubernetes: if running in-cluster, leave KUBECONFIG unset;
        # the client will auto-detect the service-account token.
        self.kubeconfig_path: Optional[str] = os.environ.get("KUBECONFIG")
        self.kube_context: Optional[str] = os.environ.get("KUBE_CONTEXT")

        # Safety: maximum lines returned by get_pod_logs
        self.max_log_tail_lines: int = int(os.environ.get("MAX_LOG_TAIL_LINES", "500"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()


# ---------------------------------------------------------------------------
# Kubernetes client factory
# ---------------------------------------------------------------------------


def get_k8s_clients() -> tuple:
    """
    Build and return a tuple of (CoreV1Api, AppsV1Api) Kubernetes clients.

    Attempts in-cluster config first (for pods running inside K8s),
    falls back to kubeconfig for local development.

    Returns
    -------
    tuple[kubernetes.client.CoreV1Api, kubernetes.client.AppsV1Api]
    """
    try:
        from kubernetes import client, config  # type: ignore

        settings = get_settings()

        try:
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes config.")
        except Exception:
            config.load_kube_config(
                config_file=settings.kubeconfig_path,
                context=settings.kube_context,
            )
            logger.info("Using kubeconfig: %s", settings.kubeconfig_path or "~/.kube/config")

        return client.CoreV1Api(), client.AppsV1Api()

    except ImportError as exc:
        raise RuntimeError(
            "kubernetes package not installed. Run: pip install kubernetes"
        ) from exc


def get_custom_objects_client():
    """
    Return a Kubernetes CustomObjectsApi client.

    Used for CRDs and operator-managed resources.
    """
    from kubernetes import client  # type: ignore

    get_k8s_clients()  # ensures config is loaded
    return client.CustomObjectsApi()
