# Adversarial Robustness Benchmarking Service

An asynchronous HTTP service that exposes an adversarial-robustness benchmarking
engine as an API. Submit a benchmark, get a job id back immediately, poll for
status, collect structured results when it finishes.

It runs entirely on your machine: `docker compose up` brings up the API, a
worker, PostgreSQL and Redis, and `./scripts/demo.sh` drives a benchmark through
the whole lifecycle.

The benchmarking library is the [original framework in this repository](docs/framework.md):
attacks (FGSM, PGD, C&W, L-BFGS, ensemble) run against a pool of pretrained
Fashion-MNIST classifiers, scored by how much accuracy each model retains under
attack. **This project is not about the ML.** It is about everything around it —
the job lifecycle, the queue, the persistence, the failure modes, the threat
model.
---

## Contents

- [Quickstart](#quickstart)
- [API reference](#api-reference)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Security considerations](#security-considerations)
- [Testing](#testing)
- [Development](#development)
- [Limitations and what I would do next](#limitations-and-what-i-would-do-next)
- [Project layout](#project-layout)

---

## Quickstart

Docker is the only prerequisite. Clone, start, run a benchmark:

```bash
git clone https://github.com/billu2006/Adversarial-Project.git
cd Adversarial-Project

docker compose up --build     # or: make up
./scripts/demo.sh             # or: make demo   (in a second terminal)
```

`demo.sh` submits a real benchmark, replays the `Idempotency-Key` to show the
same job comes back, asks for results too early to show the `409`, polls until
the worker finishes, and prints the scores. It needs nothing but `bash`, `curl`
and `python3`.

<details>
<summary>What <code>docker compose up</code> starts</summary>

Four containers — `api`, `worker`, `postgres`, `redis` — plus a one-shot
`migrate` step the API waits on, so a clean clone gets a migrated database
without anyone running Alembic by hand. The API is on
<http://localhost:8000>, with interactive docs at
<http://localhost:8000/docs>.

The first build takes a few minutes and downloads ~200MB of CPU-only PyTorch
for the worker image; later starts are seconds. The worker downloads the
Fashion-MNIST test split (~30MB) the first time it runs a job, and caches it in
a named volume.

</details>

### Doing it by hand

```bash
export KEY="X-API-Key: local-development-key"

# What can I ask for?
curl -H "$KEY" localhost:8000/v1/models
curl -H "$KEY" localhost:8000/v1/attacks

# Submit a benchmark. Returns immediately with 202.
curl -X POST localhost:8000/v1/jobs \
     -H "$KEY" -H "Content-Type: application/json" \
     -H "Idempotency-Key: $(uuidgen)" \
     -d '{"model_name": "fmnist-mlp-defender-0",
          "attacks": ["fgsm", "pgd"],
          "epsilon": 0.1,
          "max_iterations": 20}'

# Poll. queued -> running -> succeeded
curl -H "$KEY" localhost:8000/v1/jobs/<job_id>

# Collect. 409 until the job has succeeded.
curl -H "$KEY" localhost:8000/v1/jobs/<job_id>/results
```

### Watching it work

```bash
make logs                              # structured JSON logs from api + worker
make scale                             # three workers draining the same queue
docker compose exec postgres \
  psql -U benchmark -d benchmark \
  -c "SELECT id, status, model_name FROM jobs ORDER BY created_at DESC LIMIT 5;"
```

Submitting several jobs at once and watching them move through `queued` →
`running` → `succeeded` is the clearest demonstration of what the project
actually does. `make down` stops everything and removes the volumes.

---

## API reference

Every route below `/v1` requires `X-API-Key`. `/healthz` does not.

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/v1/jobs` | Submit a benchmark. **202 Accepted** with the job resource. Honours `Idempotency-Key` |
| `GET` | `/v1/jobs/{id}` | Status and timestamps. 404 if unknown |
| `GET` | `/v1/jobs/{id}/results` | Results. **409 Conflict** until the job has succeeded |
| `GET` | `/v1/jobs?status=&limit=&cursor=` | Cursor-paginated listing, newest first |
| `DELETE` | `/v1/jobs/{id}` | Cancel. Only valid from `queued` |
| `GET` | `/v1/models` | The model whitelist, with per-deployment availability |
| `GET` | `/v1/attacks` | Available attacks and the limits that apply to them |
| `GET` | `/healthz` | Liveness — actually checks Postgres and Redis |

**Submission**

```http
POST /v1/jobs
X-API-Key: local-development-key
Idempotency-Key: 8f14e45f-ea1a-4e5e-9c2b-1d3b7c6a9e01

{ "model_name": "fmnist-mlp-defender-0",
  "attacks": ["fgsm", "pgd", "cw"],
  "epsilon": 0.1,
  "max_iterations": 20 }
```

**202 Accepted** (`Location: /v1/jobs/3f9a...`)

```json
{ "job_id": "3f9a...", "status": "queued",
  "model_name": "fmnist-mlp-defender-0",
  "attacks": ["fgsm", "pgd", "cw"],
  "epsilon": 0.1, "max_iterations": 20,
  "created_at": "2026-09-06T10:14:22Z",
  "started_at": null, "finished_at": null,
  "links": { "self": "/v1/jobs/3f9a...", "results": "/v1/jobs/3f9a.../results" } }
```

**Results — 200**

```json
{ "job_id": "3f9a...", "status": "succeeded",
  "model_name": "fmnist-mlp-defender-0", "epsilon": 0.1, "max_iterations": 20,
  "results": [
    { "attack_name": "fgsm", "robust_accuracy": 0.4823, "mean_nll": 1.732104,
      "perturbation_norm": 0.1, "runtime_ms": 1840 },
    { "attack_name": "pgd", "robust_accuracy": 0.3211, "mean_nll": 2.914803,
      "perturbation_norm": 0.1, "runtime_ms": 21406 }
  ],
  "defence_score": 0.4017 }
```

**Errors** — one envelope on every failure path, never a bare 500:

```json
{ "error": { "code": "unsupported_model",
             "message": "Model 'resnet18-cifar10' is not supported.",
             "details": { "supported": ["fmnist-mlp-defender-0", "..."] } } }
```

| Code | Status | When |
|---|---|---|
| `invalid_request` | 400 | Schema violation, bad cursor, limit exceeded |
| `unsupported_model` / `unsupported_attack` | 400 | Not on the whitelist; `details.supported` lists what is |
| `unauthorized` | 401 | Missing or wrong API key |
| `not_found` | 404 | No such job |
| `results_not_ready` | 409 | Job has not succeeded; `details.status` says why |
| `job_not_cancellable` | 409 | Only a `queued` job can be cancelled |
| `idempotency_key_reuse` | 409 | Key replayed with a different body |
| `capacity_exceeded` | 503 | Too many jobs queued or running |
| `internal_error` | 500 | Logged in full, never echoed to the client |

Every response carries `X-Request-ID` — inbound ones are honoured, and the id
follows the job into the worker's logs.

---

## Architecture

```
   Client
     │  POST /v1/jobs                    (202 + job_id, immediately)
     ▼
┌──────────────┐        enqueue        ┌──────────┐
│  FastAPI     │ ────────────────────► │  Redis   │
│  (api)       │                       │  queue   │
└──────┬───────┘                       └────┬─────┘
       │                                    │ consume
       │ read/write                         ▼
       │                            ┌───────────────┐
       │                            │  Worker (RQ)  │
       │                            │  runs the     │
       │                            │  benchmark    │
       │                            └───────┬───────┘
       ▼                                    │ write results
┌───────────────────────────────────────────▼──────┐
│              PostgreSQL                          │
│   jobs · job_results                             │
└──────────────────────────────────────────────────┘
```

**Stack:** Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2.0 + Alembic ·
PostgreSQL 16 · Redis + RQ · pytest · Docker Compose · GitHub Actions.

The API and the worker share the database models and the job-transition
functions (`service/jobs.py`) but nothing else. The API never imports PyTorch;
the worker never serves a request. They are separate images for that reason.

### Data model

```sql
CREATE TYPE job_status AS ENUM ('queued','running','succeeded','failed','cancelled');

CREATE TABLE jobs (
    id                  UUID PRIMARY KEY,
    idempotency_key     TEXT UNIQUE,
    request_fingerprint VARCHAR(64),
    model_name          TEXT         NOT NULL,   -- from the whitelist
    attacks             JSONB        NOT NULL,   -- ["fgsm","pgd","cw"]
    epsilon             NUMERIC(6,4) NOT NULL,
    max_iterations      INTEGER      NOT NULL DEFAULT 20,
    status              job_status   NOT NULL DEFAULT 'queued',
    error_message       TEXT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ
);

CREATE TABLE job_results (
    id                BIGSERIAL PRIMARY KEY,
    job_id            UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attack_name       TEXT NOT NULL,
    robust_accuracy   NUMERIC(5,4) NOT NULL,
    mean_nll          NUMERIC(10,6),
    perturbation_norm NUMERIC(8,6),
    runtime_ms        INTEGER NOT NULL,
    UNIQUE (job_id, attack_name)
);

CREATE INDEX idx_jobs_status_created ON jobs (status, created_at DESC);
CREATE INDEX idx_results_job         ON job_results (job_id);
```

### Job lifecycle

```
                    ┌──────────► cancelled        (DELETE, only from queued)
                    │
  POST ──► queued ──┼──► running ──┬──► succeeded  (results written)
                    │              │
                    │              └──► failed     (engine error, timeout,
                    │                               or reaped after a crash)
                    └──► failed                    (enqueue failed)
```

A job only ever leaves `running` in a terminal state. That is the property the
whole design protects, because a client polling a status that never changes has
no way to recover.

---

## Design decisions

### Why a job queue rather than a synchronous endpoint

A benchmark is minutes of CPU. A synchronous endpoint would hit the load
proxy's idle timeout (60 seconds is a common default), hold a worker thread for
the duration, lose all its work if the client disconnected, and give the client
no way to ask "is it still going?". Accepting the job, returning a `202` with a
job id, and running the work elsewhere fixes all four at once. The cost is a
polling contract and a state machine — which is exactly the part worth building.

### Why idempotency keys, and why enforced at the database layer

A client that times out and retries must not get two benchmarks. The obvious
implementation is "look up the key; if absent, insert" — and that is wrong,
because two concurrent retries both find it absent and both insert. Under a
single client it looks fine; under the retry storm it was written for, it
duplicates.

So the constraint lives in the schema (`idempotency_key TEXT UNIQUE`), and
`create_job` inserts optimistically and catches the `IntegrityError`:

```python
try:
    session.commit()
except IntegrityError:
    session.rollback()
    existing = session.scalar(select(Job).where(Job.idempotency_key == key))
    ...
    return existing, True  # the original job, not a new one
```

The loser of the race reads the winner's job. There is no window in which both
can win, because the database is the arbiter.

One refinement: the row also stores a SHA-256 fingerprint of the submission. A
key replayed with a *different* body is a client bug, not a retry — returning
the original job would be a lie about what was run, so it gets a `409
idempotency_key_reuse`. This is the same pattern payment APIs use, and for the
same reason.

A replay returns `200 OK` with `Idempotency-Replayed: true` rather than `202`,
so a client can tell "I created this" from "this already existed".

### Why `NUMERIC` over `FLOAT`

Robust accuracy *is* the product. Two clients comparing scores, or a regression
test asserting a benchmark has not drifted, need the value that went in to be
the value that comes out. Binary floats do not round-trip predictably through
JSON and SQL, and `0.4823` stored as a float compares unequal to `0.4823` parsed
from JSON often enough to matter. `NUMERIC(5,4)` stores exactly four decimal
places of a fraction, which is the precision the measurement actually has.

### Why that composite index

```sql
CREATE INDEX idx_jobs_status_created ON jobs (status, created_at DESC);
```

Two queries need serving: "the newest jobs with status X" (the listing endpoint)
and "the oldest queued job" (operational triage). Both filter on `status` by
equality and then order by `created_at`. A B-tree can use a leading column for
equality and the next for ordering — so `(status, created_at)` serves both with
no sort step. Reversed, `(created_at, status)` could not: it would have to scan
every row in the time range and filter, because `status` would be behind an
unbounded range predicate.

### Why cursor pagination rather than `OFFSET`

Jobs are created while a client pages through the list. With `OFFSET`, a new job
at the head shifts every subsequent page by one, so the client silently sees a
duplicate and misses a row. The cursor is the sort key of the last row seen
(`created_at`, `id`, base64-encoded), and the next page is everything strictly
after it — stable under concurrent inserts, and it does not get slower as you
page deeper.

### Why the model whitelist

See [Security considerations](#security-considerations). Short version:
`torch.load()` on an untrusted file is remote code execution.

### Why the worker is a separate container, not a thread

A background thread inside the API process would be simpler and would be wrong.
The benchmark is CPU-bound Python: it would contend with request handling for
the GIL, so polling `GET /v1/jobs/{id}` would get slower exactly while a job was
running. Scaling would mean scaling the API to add compute; a crash in the
engine would take the API down with it; and `make scale` would be impossible,
because two API replicas would each run their own uncoordinated workers.

Separate processes coordinated through a queue cost one Redis container and buy
independent scaling (`docker compose up --scale worker=3` needs no code change),
an isolated failure domain, and a worker image that can carry PyTorch while the
API image does not.

It also means the local stack is the same shape as a deployed one. Running the
worker as a queue consumer is the part that a serverless platform would push
back on — Cloud Run scales to zero and drives instances by request, so a
long-lived process blocked on `BRPOP` is the wrong shape for it, and the answer
would be Cloud Run Jobs, a small always-on VM, or switching the `JobQueue`
implementation to a push-based one. That protocol exists so the choice stays
open. It is out of scope here: this is a local project by design.

### Why the API image has no PyTorch in it

The catalogue the API validates against (`benchmark/catalog.py`) is plain
dataclasses — model names, attack names, descriptions, limits. The
implementations that need torch live next to it in `attacks.py`, `models.py` and
`engine.py`, and the package's `__init__` resolves those lazily. So the API can
answer "is `fgsm` a valid attack?" and render `/v1/attacks` without importing a
tensor library. Its image is roughly 800MB smaller than the worker's and starts
in a fraction of the time, which is the difference between `docker compose up`
being usable and being annoying.

That is an invisible constraint — a stray import in a router would break the
deployed API while every local test still passed, because a development machine
has torch installed. So `tests/test_api_import_graph.py` blocks `torch` at the
import system level and builds the app, and fails if anything reaches for it.

### Why `/v1/` from the first commit

Retrofitting a version onto URLs that clients already call is a migration;
adding `/v2` next to an existing `/v1` is a routing change. The cost of the
prefix on day one is four characters.

### Why the tests can run on SQLite

`pytest` on a clean clone should not require a Docker daemon. The ORM uses two
small type decorators (`GUID`, `JSONColumn`) so the models render as
`uuid`/`jsonb` on Postgres and as portable equivalents elsewhere. The trade-off
is real — SQLite will not catch a Postgres-specific mistake — so CI runs the
same suite against a real PostgreSQL 16 service container, *and* applies the
migration to an empty database, so the two paths cannot drift silently.

---

## Security considerations

**The threat: `torch.load()` is remote code execution.** PyTorch checkpoints are
Python pickles, and unpickling runs arbitrary code by design — a malicious
`.pt` file needs no exploit, only `__reduce__`. A benchmarking service that
accepted model uploads and called `torch.load()` on them would hand any user a
shell on the worker.

**The design response, in order of preference:**

1. **No arbitrary weight uploads.** A request names a model from a whitelist
   (`benchmark/models.py`); the whitelist maps that name to a file we shipped.
   This removes the attack surface rather than trying to sanitise it. It is also
   why `POST /v1/jobs` takes `model_name` and not a URL or a file.
2. **`weights_only=True` on every load**, so even a tampered *local* checkpoint
   cannot execute code. Defence in depth: the whitelist should make this
   unreachable, and it is there in case it does not.
3. **A containment check** that the resolved path is inside the weights
   directory, so no registry entry can escape it via `..`.
4. **The extended `.pyc` attack suite is not wired into the service.** Importing
   bytecode from disk executes it — the same trust problem in a different
   costume. It stays available to the local CLI scripts, where the operator
   already trusts the files.

If uploads were ever in scope, the route would be **safetensors only** — a
non-executable format — validated before deserialisation, never raw pickles.

**Resource limits, regardless of who is asking.** An unbounded benchmark is a
denial-of-service vector even from a well-meaning user:

| Control | Default | Why |
|---|---|---|
| `epsilon` ≤ 0.109 | schema + engine clamp | The framework's constraint is a strict `< 0.11`; a benchmark above it is meaningless |
| `max_iterations` ≤ 200 | schema + settings | The dominant term in an iterative attack's runtime |
| Attacks per job ≤ 5 | settings | Bounds the multiplier on that runtime |
| Evaluation set ≤ 2048 samples | settings | Bounds the per-attack cost |
| Job timeout 900s | RQ `job_timeout` | A hard wall-clock cap; the worker turns it into a `failed` job |
| Active jobs ≤ 50 | `capacity_exceeded` 503 | Crude backpressure, but better than an unbounded queue |
| API key on every `/v1` route | `X-API-Key` | Keeps the service from being open compute the moment it leaves localhost |

**Other choices worth naming:** both containers run as a non-root user; the
error envelope never echoes an exception message to the client (it is logged
instead, since exception text leaks connection strings and row data); the API
key comparison is constant-time; and `error_message` on a job is truncated
before it becomes client-visible.

**What is deliberately *not* defended:** there is no per-key rate limiting, no
tenancy, and no authorisation beyond the single shared key. Any key holder can
read any job. That is acceptable for something that runs on localhost and would
not be for anything exposed — see
[Limitations](#limitations-and-what-i-would-do-next).

---

## Testing

Written test-first: the five edge cases below existed as failing tests before
the endpoints did.

```bash
pytest                    # everything (SQLite, no services required)
pytest -m "not engine"    # skip the slow real-PyTorch tests
pytest -m engine          # only the real-PyTorch tests
```

| File | Covers |
|---|---|
| `test_submission.py` | 202 and the job resource, unknown model/attack, epsilon and iteration caps, duplicate attacks, unknown fields, auth, and a queue outage failing the job rather than stranding it |
| `test_idempotency.py` | Replay returns the original job and enqueues once; **two concurrent sessions with the same key produce one job**; same key + different body is a 409; two keyless submissions stay independent |
| `test_lifecycle.py` | Polling, 404s, malformed ids, **results-before-completion is a 409**, results after success (including exact `NUMERIC` round-tripping), and cancellation — including that cancel loses cleanly to a worker that claimed the job first |
| `test_worker.py` | Success writes results and timestamps; **a crashing benchmark leaves the job `failed`**, never `running`; a cancelled job is never executed; a job is claimed exactly once even on duplicate delivery; **the reaper fails jobs abandoned by a dead worker** and leaves healthy ones alone; the request id survives the queue hop |
| `test_listing.py` | Newest-first ordering, cursor pagination visiting every job exactly once, status filtering, malformed cursors |
| `test_catalog_and_health.py` | The whitelist and limits are published; `/healthz` reports each dependency and returns 503 when one is down |
| `test_api_import_graph.py` | **The API imports with PyTorch blocked at the import system** — the invisible constraint behind the two-image build |
| `test_engine.py` | The real engine against the real checkpoints: **every attack respects the epsilon budget**, only whitelisted models load, metrics are plausible, and an attack that cheats the budget fails the job instead of publishing a score |

The queue is a recording double in the API tests, which is what makes "was this
enqueued exactly once?" assertable. The benchmark engine is stubbed in the
worker tests: what is under test there is the promise that a job always reaches
a terminal state, not PyTorch.

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, the
migration against an empty Postgres, and the full suite against PostgreSQL 16
and Redis 7 service containers, then builds the API image.

---

## Development

The stack runs entirely in Docker, but the tests do not need it — they run
against SQLite with no services at all, which is what keeps the feedback loop
fast:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

make test          # or: pytest
make test-fast     # skip the slow real-PyTorch tests
make lint          # ruff check + ruff format --check
make format        # apply fixes
```

`make help` lists every target. `torch` is only needed for the `engine`-marked
tests and the CLI scripts in [`scripts/`](scripts/); the API and the rest of the
suite run without it.

<details>
<summary>Running the API and worker on the host, without Docker</summary>

Useful if you want a debugger on the worker. You need a Redis
(`brew install redis`); SQLite can stand in for Postgres.

```bash
brew install redis && redis-server &

cat > .local.env <<'ENV'
DATABASE_URL=sqlite:///./local.db
REDIS_URL=redis://localhost:6379/0
QUEUE_BACKEND=redis
API_KEY=local-development-key
JSON_LOGS=false
ENV
set -a && . ./.local.env && set +a

python -c "from service.database import engine; from service.models import Base; Base.metadata.create_all(engine)"
uvicorn service.main:app --port 8000   # terminal 1
python -m service.worker.main          # terminal 2
./scripts/demo.sh                      # terminal 3
```

**One platform note.** RQ forks a work-horse process per job, which is what the
container wants: a fresh process per benchmark and a timeout that can be
enforced by killing it. macOS will not allow it — PyTorch initialises
Objective-C state on import, and Apple's runtime aborts in any forked child that
has not `exec`'d, so the work-horse dies mid-benchmark and the job is left for
the reaper. `WORKER_CLASS=auto` (the default) therefore forks on Linux and runs
jobs in-process everywhere else. `resolve_worker_class` in
[`src/service/worker/main.py`](src/service/worker/main.py) is the whole of it.

</details>

Configuration is environment variables all the way down
([`src/service/config.py`](src/service/config.py)); Compose sets them, and
[`.env.example`](.env.example) documents them for a run outside it. There is no
`if environment == "production"` branch anywhere in the code — the only thing
the application knows about where it is running is the `ENVIRONMENT` string it
puts in a log line.

**Scope, stated deliberately.** This is a local project. There is no hosted
deployment, no Kubernetes, no autoscaling, no user accounts — a single static
API key stands in for authentication. Those are omissions, not oversights; the
engineering worth showing here is the job lifecycle, and adding a cloud bill
would not make that part better.

---

## Project layout

```
├── src/
│   ├── benchmark/            # the ML library - no HTTP, no database
│   │   ├── attacks.py        #   FGSM, PGD, C&W, L-BFGS, ensemble
│   │   ├── catalog.py        #   THE WHITELIST + attack metadata (no torch import)
│   │   ├── constants.py      #   epsilon and iteration limits
│   │   ├── data.py           #   Fashion-MNIST test-split loader
│   │   ├── engine.py         #   run_benchmark(): what a job executes
│   │   └── models.py         #   DefenderNet + guarded weight loading
│   └── service/
│       ├── main.py           #   app factory
│       ├── config.py         #   settings and every resource limit
│       ├── database.py       #   engine, session, dependency
│       ├── models.py         #   ORM: jobs, job_results
│       ├── schemas.py        #   the wire contract
│       ├── jobs.py           #   lifecycle logic (shared by API and worker)
│       ├── queue.py          #   JobQueue protocol: RQ, or inline for tests
│       ├── errors.py         #   the error envelope
│       ├── security.py       #   API key
│       ├── logging_config.py #   JSON logs + request-id propagation
│       ├── middleware.py     #   request context and access logs
│       ├── routers/          #   jobs, catalog, health
│       └── worker/           #   RQ entrypoint, the task, the reaper
├── migrations/               # Alembic
├── tests/                    # pytest
├── scripts/                  # demo.sh (end-to-end walkthrough) + CLI evaluation tools
├── notebooks/                # the original adversarial-training notebook
├── models/defenders/         # the ten pretrained checkpoints
├── docs/framework.md         # the benchmarking library, documented on its own
├── docker-compose.yml        # api · worker · postgres · redis (+ migrate)
├── Dockerfile                # two targets: api (no torch) and worker (torch)
└── Makefile                  # up · demo · logs · scale · test · lint
```

---

## Credits

Built on the adversarial robustness framework in this repository — originally a
COMP219 university assignment, extended and then wrapped in the service
described above. The library half is documented separately in
[docs/framework.md](docs/framework.md).
