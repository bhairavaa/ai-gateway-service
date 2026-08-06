# 🚀 AI Gateway Service

**A production-style API gateway that puts one unified endpoint in front of multiple LLM providers.**

Send one request shape to `/v1/chat/completions` and route it to **OpenAI, Anthropic Claude, Google Gemini, or Ollama** — with automatic fallback, streaming, per-key rate limiting, and cost tracking baked in. Built with FastAPI, async SQLAlchemy, and Redis.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white">
  <img alt="SQLAlchemy" src="https://img.shields.io/badge/SQLAlchemy-2.0%20async-D71F00?logo=sqlite&logoColor=white">
  <img alt="Redis" src="https://img.shields.io/badge/Redis-rate%20limiting-DC382D?logo=redis&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-104%20passing-brightgreen?logo=pytest&logoColor=white">
</p>

---

## 📌 At a glance

| | |
|---|---|
| **Providers** | OpenAI · Anthropic Claude · Google Gemini · Ollama |
| **API surface** | One OpenAI-Chat-Completions-shaped endpoint, JSON + SSE streaming |
| **Auth** | API-key (`X-API-Key`), hashed at rest, active/expiry/scopes |
| **Resilience** | Configurable per-request fallback chains across providers |
| **Persistence** | Async SQLAlchemy + Alembic migrations (SQLite → Postgres-ready) |
| **Rate limiting** | Redis fixed-window, per API key, fails open |
| **Testing** | 104 tests, fully mocked/in-memory — no live network in CI |
| **Deployment** | Docker + docker-compose, migrations run on container start |

## ✨ Features

- 🔀 **One API** — a single endpoint (`POST /v1/chat/completions`) routes to any of the four providers
- 🛡️ **Automatic fallback** — configurable provider/model chains; a failed primary call transparently retries the next entry
- 📡 **Streaming** — Server-Sent Events (SSE) and plain JSON share the same auth/rate-limit/fallback pipeline
- 🔑 **API-key auth** — `X-API-Key` header, SHA-256 hashed at rest, with active/inactive/expiry/scopes support
- ⏱️ **Per-key rate limiting** — Redis fixed-window limiter, fails open if Redis is unreachable
- 💰 **Cost & usage tracking** — every request's tokens, provider, model, duration, and estimated cost persisted to the DB
- 🧩 **Clean middleware pipeline** — request ID propagation, timing, structured logging, standardized error envelope
- 🗄️ **Repository pattern** — SQLite today, swappable for Postgres later with no service/route changes

## 🏗️ Architecture

```
Client
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Middleware (outer → inner)                                   │
│  ExceptionHandling → RequestID → Timing → Authentication      │
│    → RateLimit → Logging → CostTracking                       │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│  API layer            │────▶│  Business logic (services/)   │
│  routers, schemas      │     │  ChatService, FallbackExecutor│
└─────────────────────┘     │  pricing, outcome persistence │
                              └──────────────┬───────────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                    ▼                    ▼
              ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
              │  Provider layer    │ │  Database layer    │ │  Config layer      │
              │  BaseProvider ABC   │ │  repositories/     │ │  Settings (env)    │
              │  + Registry/Factory │ │  ORM models         │ │                     │
              └──────────────────┘ └──────────────────┘ └──────────────────┘
                        │
        ┌───────┬───────┼───────┬────────┐
        ▼       ▼       ▼        ▼
     OpenAI  Claude  Gemini   Ollama
```

Every provider implements one interface (`app/providers/base.py`): `generate()`, `stream()`, `list_models()`, `health_check()`. The gateway never branches on provider name — `ProviderRegistry` resolves providers dynamically from config, so adding a fifth provider is one new file + one registry entry.

## 📁 Project layout

```
app/
├── main.py              # FastAPI app factory, middleware registration order
├── config.py             # Settings (env-driven, pydantic-settings)
├── api/                   # Routers, request/response wiring, exception handlers
├── middleware/            # RequestID, Timing, Auth, RateLimit, Logging, CostTracking, ExceptionHandling
├── services/              # ChatService, FallbackExecutor, pricing, streaming, outcome persistence
├── providers/              # BaseProvider + OpenAI/Claude/Gemini/Ollama adapters + Registry
├── auth/                   # AuthBackend abstraction + API-key implementation
├── db/                     # SQLAlchemy models, session factory, repositories (+ interfaces)
└── schemas/                 # Pydantic request/response DTOs
scripts/create_api_key.py    # Bootstrap CLI — mints an API key directly via the repository
tests/
├── unit/                    # Providers, middleware, services, repositories — all mocked/in-memory
└── integration/              # Full request/response through TestClient
```

## ⚡ Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements-dev.txt

cp .env.example .env              # then fill in whichever provider API keys you have
mkdir -p data
alembic upgrade head              # creates data/gateway.db

python scripts/create_api_key.py "my-client"    # prints a raw key — save it, shown once
```

Run it:

```bash
uvicorn app.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

## 🐳 Docker

```bash
cp .env.example .env    # fill in provider keys
docker compose up --build
```

This starts the gateway (port 8000) and Redis, with the SQLite file on a named volume so it survives rebuilds. Migrations run automatically on container start.

## 🔌 API usage

All `/v1/*` routes require `X-API-Key`. `/health` does not.

**Non-streaming:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

**Streaming (SSE):**

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "provider": "openai", "model": "gpt-4o-mini", "stream": true,
    "messages": [{"role": "user", "content": "Count to five."}]
  }'
```

**Explicit fallback chain:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "provider": "openai", "model": "gpt-4o-mini",
    "fallback": [{"provider": "anthropic", "model": "claude-haiku-4-5"}],
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

If `fallback` is omitted, the gateway falls back to `DEFAULT_FALLBACK_CHAIN` from config instead of trying nothing else.

**Discovery:**

```bash
curl http://localhost:8000/v1/models -H "X-API-Key: $API_KEY"
curl http://localhost:8000/v1/health/providers -H "X-API-Key: $API_KEY"
```

## ✅ Testing

```bash
pytest
```

104 tests, no live network, Redis, or on-disk provider calls in the suite — providers are mocked, Redis is `fakeredis`, and the DB is a dedicated on-disk SQLite test file seeded with one API key (see `tests/conftest.py`).

## 🧠 Engineering decisions

A few deliberate tradeoffs, worth knowing about rather than discovering by surprise:

- **Fixed-window rate limiting**, not sliding-window: two Redis calls (`INCR` + `EXPIRE`), easy to reason about, at the cost of allowing up to ~2x the limit in a burst straddling a window boundary.
- **No mid-stream fallback**: once a provider's first SSE chunk has been sent to the client, a failure partway through surfaces as a terminal error event, not a silent switch to the next provider — you can't un-send bytes.
- **`cost_usd` is `None`, never `0`, for unmapped models** — an unpriced model should never silently look free.
- **API keys are SHA-256 hashed**, not bcrypt/argon2 — appropriate for a high-entropy machine credential (not a human password vulnerable to dictionary attack against a leaked hash).
- **No admin HTTP surface for key management** — `scripts/create_api_key.py` talks to the repository directly. Anyone who can run it already has DB/filesystem access, so there's no separate auth-for-auth problem to solve.
- **Auth is API-key only today**, but sits behind an `AuthBackend` interface so JWT/OAuth can be added later as a second implementation, without touching routes or other middleware.

## 🛠️ Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Validation / config | Pydantic v2, pydantic-settings |
| Database | SQLAlchemy 2.0 (async) + Alembic migrations |
| Cache / rate limiting | Redis |
| LLM integrations | LangChain (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-ollama`) |
| Testing | pytest, pytest-asyncio, fakeredis |
| Containerization | Docker, docker-compose |

## 📄 License

Not yet licensed for reuse — all rights reserved by the author unless a `LICENSE` file is added.
