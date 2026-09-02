"""Adversarial attacks, parametrised so the API can expose them as job options.

The original repository hard-coded epsilon and the iteration count into each
attack (``fgsm_attack``/``pgd5_attack``/``pgd20_attack``). A benchmarking
*service* has to take both from the request, so every attack here shares one
signature:

    fn(model, X, y, *, epsilon, max_iterations, device) -> Tensor

``X`` is a flattened batch of shape ``(N, 784)`` in ``[0, 1]``; the returned
adversarial batch has the same shape and is clamped back into the valid image
range and into the L-inf ball of radius ``epsilon``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from benchmark.catalog import ATTACK_CATALOG, AttackInfo
from benchmark.constants import MAX_EPSILON

AttackFn = Callable[..., Tensor]


def _clip(x_adv: Tensor, x_orig: Tensor, epsilon: float) -> Tensor:
    """Project ``x_adv`` back into the epsilon-ball around ``x_orig`` and [0, 1].

    Every attack funnels through this so the epsilon constraint is enforced in
    exactly one place rather than being re-derived (and occasionally fumbled) by
    each implementation.
    """
    delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


def _safe_epsilon(epsilon: float) -> float:
    return float(min(epsilon, MAX_EPSILON))


def fgsm(model: nn.Module, X: Tensor, y: Tensor, *, epsilon: float, **_: object) -> Tensor:
    """Fast Gradient Sign Method: one gradient step of size epsilon."""
    epsilon = _safe_epsilon(epsilon)
    x_adv = X.clone().detach().requires_grad_(True)

    loss = F.nll_loss(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    with torch.no_grad():
        return _clip(x_adv + epsilon * x_adv.grad.sign(), X, epsilon)


def pgd(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    *,
    epsilon: float,
    max_iterations: int = 20,
    **_: object,
) -> Tensor:
    """Projected Gradient Descent: the iterative form of FGSM.

    The step size follows the usual ``2.5 * epsilon / iterations`` heuristic so
    the attack can still traverse the ball whatever iteration count the caller
    asked for, instead of the fixed alpha the original scripts used.
    """
    epsilon = _safe_epsilon(epsilon)
    steps = max(1, int(max_iterations))
    alpha = 2.5 * epsilon / steps

    # Random start inside the ball - a deterministic start makes PGD easy for a
    # defence to overfit against.
    x_adv = _clip(X + torch.empty_like(X).uniform_(-epsilon, epsilon), X, epsilon)

    for _step in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = F.nll_loss(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = _clip(x_adv + alpha * x_adv.grad.sign(), X, epsilon)

    return x_adv.detach()


def carlini_wagner(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    *,
    epsilon: float,
    max_iterations: int = 100,
    confidence: float = 0.0,
    c: float = 0.5,
    learning_rate: float = 0.01,
    **_: object,
) -> Tensor:
    """Carlini & Wagner style attack, optimised in tanh space.

    Slower than PGD but minimises the margin directly rather than the loss, so
    it tends to find adversarial examples PGD misses.
    """
    epsilon = _safe_epsilon(epsilon)
    steps = max(1, int(max_iterations))

    w = torch.zeros_like(X, requires_grad=True)
    optimizer = optim.Adam([w], lr=learning_rate)

    for _step in range(steps):
        # tanh reparametrisation keeps the iterate inside the image box without
        # a projection step fighting the optimiser.
        x_adv = _clip(0.5 * (torch.tanh(w) + 1.0), X, epsilon)
        outputs = model(x_adv)

        true_logit = outputs.gather(1, y.unsqueeze(1)).squeeze(1)
        others = outputs.clone()
        others.scatter_(1, y.unsqueeze(1), -float("inf"))
        best_other = others.max(1)[0]

        margin = torch.clamp(true_logit - best_other + confidence, min=0.0).mean()
        distortion = ((x_adv - X) ** 2).sum(dim=1).mean()
        loss = margin + c * distortion

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return _clip(0.5 * (torch.tanh(w) + 1.0), X, epsilon)


def lbfgs(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    *,
    epsilon: float,
    max_iterations: int = 50,
    l2_reg: float = 1e-5,
    learning_rate: float = 1.0,
    **_: object,
) -> Tensor:
    """L-BFGS attack: second-order optimisation of the negative loss.

    Strong but expensive; capped by ``max_iterations`` like everything else. The
    optimiser can fail to converge on degenerate batches, in which case we fall
    back to whatever iterate it reached rather than failing the whole job.
    """
    epsilon = _safe_epsilon(epsilon)
    x_orig = X.detach().clone()
    x_adv_param = nn.Parameter(x_orig.clone())
    optimizer = optim.LBFGS(
        [x_adv_param],
        lr=learning_rate,
        max_iter=max(1, int(max_iterations)),
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_adv_param)
        # Negative NLL because we are maximising the model's loss.
        loss = -F.nll_loss(outputs, y) + l2_reg * (x_adv_param - x_orig).pow(2).sum(dim=1).mean()
        loss.backward()
        return loss

    # Line search can blow up on flat regions; the iterate reached so far is
    # still a valid (if weaker) adversarial candidate once clipped.
    with contextlib.suppress(RuntimeError):
        optimizer.step(closure)

    return _clip(x_adv_param.detach(), x_orig, epsilon)


def ensemble(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    *,
    epsilon: float,
    max_iterations: int = 50,
    device: torch.device | None = None,
    **_: object,
) -> Tensor:
    """Run several attacks and keep, per batch, whichever hurt the model most.

    This is the strongest option and the one the original notebook submitted for
    grading. Its cost is roughly the sum of its members', which is why the
    service treats it as a separate attack rather than a free upgrade.
    """
    epsilon = _safe_epsilon(epsilon)
    candidates = [
        pgd(model, X, y, epsilon=epsilon, max_iterations=max_iterations),
        carlini_wagner(model, X, y, epsilon=epsilon, max_iterations=max_iterations),
        lbfgs(model, X, y, epsilon=epsilon, max_iterations=max_iterations),
    ]

    best, best_loss = X, -float("inf")
    with torch.no_grad():
        for candidate in candidates:
            loss = F.nll_loss(model(candidate), y).item()
            if loss > best_loss:
                best, best_loss = candidate, loss
    return best


#: Attack name -> implementation. The *metadata* for each of these lives in
#: benchmark.catalog, which the API imports without torch; this mapping is what
#: binds a name to code, and it is checked against the catalogue below so the
#: two can never drift.
ATTACK_FUNCTIONS: dict[str, AttackFn] = {
    "fgsm": fgsm,
    "pgd": pgd,
    "cw": carlini_wagner,
    "lbfgs": lbfgs,
    "ensemble": ensemble,
}


@dataclass(frozen=True)
class AttackSpec:
    """An attack's published description together with its implementation."""

    info: AttackInfo
    fn: AttackFn = field(repr=False)

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def description(self) -> str:
        return self.info.description

    @property
    def uses_iterations(self) -> bool:
        return self.info.uses_iterations

    @property
    def relative_cost(self) -> str:
        return self.info.relative_cost


if set(ATTACK_FUNCTIONS) != set(ATTACK_CATALOG):
    # A catalogued attack with no implementation would be advertised by the API
    # and then fail every job that asked for it; an implementation missing from
    # the catalogue would be unreachable. Fail at import instead.
    raise RuntimeError(
        "Attack catalogue and implementations disagree: "
        f"{set(ATTACK_CATALOG) ^ set(ATTACK_FUNCTIONS)}"
    )

ATTACKS: dict[str, AttackSpec] = {
    name: AttackSpec(info=info, fn=ATTACK_FUNCTIONS[name]) for name, info in ATTACK_CATALOG.items()
}


def list_attacks() -> list[AttackSpec]:
    return list(ATTACKS.values())


def get_attack(name: str) -> AttackSpec:
    try:
        return ATTACKS[name]
    except KeyError as exc:  # pragma: no cover - the API validates before calling
        raise KeyError(f"Unknown attack: {name!r}") from exc
