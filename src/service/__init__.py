"""HTTP service wrapping the adversarial-robustness benchmarking engine.

Benchmarking a model takes minutes, so the API never runs one inline: it
validates the request, persists a job, enqueues it and returns ``202 Accepted``
with a job id. A worker process drains the queue and writes results back. The
client polls.
"""

__version__ = "0.1.0"
