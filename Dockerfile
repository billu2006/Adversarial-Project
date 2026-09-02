# Two images from one file. They share the application layer but not their
# dependencies: only the worker needs PyTorch, and keeping ~800MB of it out of
# the API image keeps it small and quick to start, which is most of what makes
# `docker compose up` pleasant to sit through.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency metadata first, so a source-only change reuses the install layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run as a non-root user. The worker executes model code; if anything ever gets
# out of the sandbox, it should not land as root.
RUN useradd --create-home --uid 10001 appuser
COPY alembic.ini ./
COPY migrations ./migrations


# --- API ------------------------------------------------------------------- #
FROM base AS api

USER appuser
EXPOSE 8080
# One uvicorn worker per container: concurrency comes from running more
# containers, not more processes inside one.
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]


# --- Worker ---------------------------------------------------------------- #
FROM base AS worker

# The CPU wheel index, explicitly. The default index would pull the CUDA build:
# several gigabytes of driver payload nothing here can use.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
 && pip install --no-cache-dir numpy

# Weights are baked into the worker image rather than fetched at runtime: they
# are the trust boundary, and an image is easier to audit than a download.
COPY models ./models
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser
ENV BENCHMARK_WEIGHTS_DIR=/app/models/defenders \
    BENCHMARK_DATA_DIR=/app/data
CMD ["python", "-m", "service.worker.main"]
