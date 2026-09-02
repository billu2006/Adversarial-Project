"""The queue consumer: where benchmarks actually run.

Separate process, separate container, separate failure domain. The API stays
responsive while a job burns CPU for minutes, and a worker that dies takes
nothing with it that the reaper cannot clean up.
"""
