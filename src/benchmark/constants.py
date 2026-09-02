"""Constants shared by the engine and the service.

Kept free of PyTorch imports so the API process can validate requests against
the same limits the worker enforces without pulling in torch.
"""

# The original assignment constraint: an attack's L-inf perturbation must be
# STRICTLY less than 0.11. We advertise 0.109 as the maximum accepted epsilon so
# that a request sitting exactly on the boundary can never produce a benchmark
# that violates the constraint after floating-point rounding.
MAX_EPSILON = 0.109

# Hard ceiling on iterative attacks. An unbounded iteration count is a
# denial-of-service vector (see "Security considerations" in the README), so the
# API rejects anything larger rather than letting the worker discover it.
MAX_ITERATIONS_LIMIT = 200

# Fashion-MNIST images are 28x28 greyscale and the defender networks are
# fully-connected, so every batch is flattened to this many features.
INPUT_FEATURES = 28 * 28
NUM_CLASSES = 10
