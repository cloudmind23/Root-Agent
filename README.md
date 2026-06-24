<div align="center">

# 🛸 Root Agent

### Autonomous Kubernetes Healing Agent

*Powered by Claude · Orchestrated by LangGraph · Built for Production SRE*

---

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/Claude-claude--sonnet--4--6-D97706?style=flat-square)](https://anthropic.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-4F46E5?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Redis](https://img.shields.io/badge/Redis-5.x-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 🌌 What Is Root Agent?

Root Agent is an **autonomous AI agent** that monitors your Kubernetes cluster, diagnoses failures, and proposes (or applies) remediations — all with a human-in-the-loop approval gate before any destructive action is taken.

When an alert fires — `CrashLoopBackOff`, `OOMKilled`, `ImagePullBackOff`, scheduling failures — Root Agent kicks off an investigation loop: it fetches logs, describes resources, pulls cluster events, searches internal runbooks, and scours the web for known fixes. It then proposes a precise `kubectl patch` and waits for operator sign-off before touching anything.

> **Think of it as a senior SRE that never sleeps, never misses a log line, and always asks before making changes.**

---

## 🪐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Root Agent                           │
│                                                                 │
│  Alert / Webhook                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐    POST /agent/analyze                             │
│  │ FastAPI │ ──────────────────────► Redis (run state)          │
│  │  :8082  │                              │                     │
│  └────┬────┘                              │                     │
│       │ asyncio.create_task               ▼                     │
│       │                        ┌─────────────────────┐         │
│       │                        │   LangGraph Loop    │         │
│       │                        │                     │         │
│       │                        │  ┌───────────────┐  │         │
│       │                        │  │   llm_node    │  │         │
│       │                        │  │  (Claude API) │  │         │
│       │                        │  └──────┬────────┘  │         │
│       │                        │         │ tool_use  │         │
│       │                        │  ┌──────▼────────┐  │         │
│       │                        │  │  tool_node    │  │         │
│       │                        │  │  (parallel)   │  │         │
│       │                        │  └──────┬────────┘  │         │
│       │                        │         │            │         │
│       │                        │  ┌──────▼────────┐  │         │
│       │                        │  │ approval_node │  │         │
│       │                        │  │  (HITL gate)  │  │         │
│       │                        │  └───────────────┘  │         │
│       │                        └─────────────────────┘         │
│       │                                                         │
│  ┌────▼────┐    POST /agent/approve/{id}                        │
│  │Operator │ ──────────────────────► apply_fix                  │
│  │Dashboard│                         (kubectl patch / restart)  │
│  │  /ui    │                                                     │
│  └─────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 🧩 Components

| Layer | Technology | Role |
|---|---|---|
| **API** | FastAPI + uvicorn | HTTP endpoints, background task dispatch |
| **Agent Loop** | LangGraph `StateGraph` | Multi-step reasoning orchestration |
| **LLM** | Anthropic `claude-sonnet-4-6` | Diagnosis, tool selection, fix proposals |
| **Tools** | Python async executors | kubectl, K8s API, RAG, Slack, web search |
| **State** | Redis (async) | Run persistence, history index, TTL |
| **Observability** | Prometheus + redis-exporter | Metrics scraping and alerting |
| **UI** | Vanilla HTML/JS | Operator dashboard at `/ui` |

---

## 🔭 Features

### 🤖 Autonomous Investigation
Root Agent follows a systematic SRE playbook on every alert:

1. **Fetch logs** — current and previous container logs with configurable tail depth
2. **Pull events** — cluster-wide or namespace-scoped warning events
3. **Describe resources** — full spec/status for pods, deployments, statefulsets, nodes
4. **Search runbooks** — RAG-powered lookup against your internal knowledge base
5. **Web search** — fallback for unknown third-party errors (Serper API)
6. **Propose fix** — conservative, targeted kubectl patch or pod restart
7. **Notify** — Slack alert at key milestones

### 🛡️ Human-in-the-Loop Safety Gate
No fix ever touches the cluster without explicit operator approval. The `apply_fix` tool carries a `human_approved` flag that the LLM is **explicitly instructed never to set itself**. The API enforces this at runtime as a secondary guard.

```
Agent proposes fix  →  status: awaiting_approval
                              │
              ┌───────────────┴──────────────┐
              │                              │
    POST /approve {approved: true}  POST /approve {approved: false}
              │                              │
         Fix applied                   Run cancelled
```

### 🗂️ Full Audit Trail
Every run stores a complete trace: every tool call, every result, every LLM step, timestamps, approver identity, and the final diagnosis. Persisted to Redis with a 7-day TTL.

### 📊 Live Operator Dashboard
Built-in web UI at `/ui` with:
- Real-time run list with status badges and auto-polling
- Cluster health overview (pod counts, restart rates, node status) via Prometheus
- Expandable tool call trace with inputs and outputs
- One-click approve / reject buttons
- In-browser alert trigger form

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (local or Docker)
- `kubectl` configured with cluster access
- Anthropic API key

### 1. Clone & Install

```bash
git clone https://github.com/your-org/root-agent
cd root-agent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...          # Required
REDIS_URL=redis://localhost:6379      # Required
RAG_API_URL=http://rag-api:8000       # Optional — runbook search
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # Optional
SERPER_API_KEY=...                    # Optional — web search fallback
```

### 3. Start Redis

```bash
# Docker
docker run -d -p 6379:6379 redis:7-alpine

# or Homebrew
brew services start redis
```

### 4. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8082 --reload
```

### 5. Open the Dashboard

```
http://localhost:8082/ui
```

---

## 🌠 Full Stack (Docker Compose)

```bash
docker-compose up -d
```

Services started:
- `agent` — Root Agent API on `:8082`
- `redis` — State store on `:6379`
- `redis-exporter` — Prometheus exporter on `:9121`

---

## 🔌 API Reference

### `POST /agent/analyze`
Trigger a new autonomous analysis run.

```bash
curl -X POST http://localhost:8082/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "alert_summary": "Pod api-server-0 in production is CrashLoopBackOff with OOMKilled exit code",
    "namespace": "production",
    "cluster": "prod-us-east-1"
  }'
```

Returns `202 Accepted` with the `AgentRun` record. Poll `GET /agent/runs/{run_id}` to track progress.

---

### `GET /agent/runs/{run_id}`
Fetch the full run record including tool trace, diagnosis, and current status.

**Run statuses:**

| Status | Meaning |
|---|---|
| `pending` | Created, agent loop not yet started |
| `running` | Agent is actively investigating |
| `awaiting_approval` | Fix proposed, waiting for operator sign-off |
| `completed` | Investigation finished successfully |
| `failed` | Unrecoverable error during the loop |
| `cancelled` | Operator rejected the proposed fix |

---

### `POST /agent/approve/{run_id}`
Approve or reject a pending fix.

```bash
# Approve
curl -X POST http://localhost:8082/agent/approve/RUN_ID \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "approver": "alice@company.com"}'

# Reject
curl -X POST http://localhost:8082/agent/approve/RUN_ID \
  -H "Content-Type: application/json" \
  -d '{"approved": false, "approver": "alice@company.com"}'
```

---

### `GET /agent/history`
List recent runs, newest first. Supports pagination and filtering.

```
GET /agent/history?limit=20&offset=0&status=awaiting_approval&cluster=prod-us-east-1
```

---

## 🛠️ Tool Catalog

| Tool | Purpose |
|---|---|
| `get_pod_logs` | Fetch current or previous container logs (tail configurable) |
| `describe_resource` | `kubectl describe` any K8s resource type |
| `get_events` | Cluster events filtered by namespace, resource, or type |
| `search_runbooks` | RAG query against internal runbook API |
| `apply_fix` | Patch a resource or restart a pod (requires `human_approved: true`) |
| `notify_slack` | Post alert message to configured Slack webhook |
| `web_search` | Search the web for unknown error codes (Serper API) |

All tools run concurrently per LangGraph iteration via `asyncio.gather`. The agent caps at **15 LLM iterations** as a safety circuit breaker.

---

## 📁 Project Structure

```
root-agent/
├── app/
│   ├── main.py              # FastAPI app, routes, background task
│   ├── models.py            # Pydantic models (AgentRun, ToolCall, etc.)
│   ├── agent/
│   │   └── graph.py         # LangGraph StateGraph, nodes, routing
│   ├── tools/
│   │   ├── definitions.py   # Claude API tool schemas (JSON)
│   │   └── executor.py      # Async tool implementations
│   ├── core/
│   │   └── config.py        # Settings, K8s client factory
│   └── static/
│       └── dashboard.html   # Operator UI
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_models.py       # Pydantic model tests
│   ├── test_executor.py     # Tool executor tests (mocked K8s)
│   ├── test_api.py          # FastAPI endpoint tests
│   └── test_graph.py        # LangGraph node/routing tests
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
└── .env.example
```

---

## 🧪 Tests

```bash
pip install pytest pytest-asyncio httpx
pytest                    # run all 110 tests
pytest -v tests/test_graph.py   # specific module
pytest --tb=short         # compact output
```

All external dependencies (Kubernetes API, Anthropic, Redis, HTTP) are mocked. Tests run fully offline.

---

## 📡 Observability

Prometheus metrics exposed at `GET /metrics`:

| Metric | Type | Description |
|---|---|---|
| `root_agent_runs_total` | Counter | Total runs started, by cluster and status |
| `root_agent_run_duration_seconds` | Histogram | Wall-clock run duration by cluster |

Scrape config: `prometheus.yml` includes the agent, Redis exporter, and self-scrape.

---

## 🗺️ Roadmap

The current build is v0.1 — the foundation is solid. Here's what's next:

- [ ] **🔭 RAG runbook service** — vector search over Confluence / Notion / Markdown docs
- [ ] **📟 Alertmanager webhook** — native Prometheus Alertmanager receiver
- [ ] **🔄 Multi-cluster support** — agent runs scoped to named kubeconfig contexts
- [ ] **📈 Cost tracking** — per-run Claude token usage and estimated cost
- [ ] **🔐 Auth** — API key or OAuth2 gate on the dashboard and API
- [ ] **📦 Helm chart** — deploy Root Agent itself into the cluster it monitors
- [ ] **🧠 Memory** — vector store of past diagnoses to accelerate future investigations
- [ ] **🪝 Integrations** — PagerDuty, OpsGenie, and Datadog alert sources

---

## 🌍 Contributing

Pull requests welcome. Please open an issue first for significant changes.

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Add tests for new behavior
4. Ensure `pytest` passes: `pytest`
5. Open a PR with a clear description

---

## ⚖️ License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

*Built with 🤖 Claude + ☸️ Kubernetes + ❤️ for on-call engineers everywhere*

</div>
