"""Synthetic system #3: partial routing collapse under traffic shift.

Twenty workers: one generalist and nineteen specialists, each uniquely right on
its own slice of tasks. The reward is binary -- one point if the chosen worker
succeeds. There is no partial credit anywhere; this collapse has a different
cause.

Phase 1: every slice is equally common and every specialist learns its slice.
Phase 2: the traffic distribution shifts and some slices become rare. Those
specialists stop appearing in the gradient, the parameters they relied on get
overwritten by the traffic that remains, and their routing decays. Some die.

The point of the design: losing three specialists out of twenty barely moves
the aggregate routing distribution. A monitor watching `max_k pi_k` is blind to
it by construction. Failure has to be defined on held-out behaviour instead.

Failure = coverage. For each specialist, c_k = P(route to k | held-out task
from k's slice), measured on a fixed evaluation set. Dead when c_k < 0.5. The
run has failed when the third specialist dies. No monitor reads this quantity.

Run:  python toy_partial_collapse.py    # -> data/partial_*.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import coverage_failure, pressure

N_WORKERS = 20                 # worker 0 is the generalist
N_SPEC = N_WORKERS - 1
D_IN, D_HIDDEN = N_WORKERS + 4, 64
STEPS, T_SHIFT = 2600, 1000
BATCH, LR = 64, 5e-3
BETA_BASE, BETA_NULL = 0.15, 0.30   # base keeps phase 1 explorable
N_EVAL = 2000
EVAL_EVERY = 10
RARE = tuple(range(1, 7))      # the six slices that become rare
P_GENERAL = 0.30               # share of traffic that is not a specialist slice
DEAD_THR, N_DEAD_FAIL = 0.5, 3
CONDITIONS = ("shift", "stable", "mild_shift", "entropy_reg")
DATA = Path(__file__).parent / "data"


class Router(nn.Module):
    """Deliberately small. Capacity pressure is what makes rare slices get
    overwritten by the traffic that remains -- the mechanism under study."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_IN, D_HIDDEN), nn.Tanh(),
                                 nn.Linear(D_HIDDEN, N_WORKERS))

    def forward(self, x):
        return self.net(x)


def slice_probs(shifted: bool, strength: float) -> np.ndarray:
    """Traffic distribution over task types 0..19 (0 = general)."""
    p = np.empty(N_WORKERS)
    p[0] = P_GENERAL
    p[1:] = (1.0 - P_GENERAL) / N_SPEC
    if shifted:
        freed = p[list(RARE)].sum() * (1.0 - strength)
        p[list(RARE)] *= strength
        keep = [k for k in range(1, N_WORKERS) if k not in RARE]
        p[keep] += freed / len(keep)
    return p / p.sum()


def sample_tasks(n, rng, p_type):
    """Task features carry a noisy one-hot of the type; 4 trailing distractors."""
    t = rng.choice(N_WORKERS, size=n, p=p_type)
    x = rng.normal(0.0, 0.35, size=(n, D_IN)).astype(np.float32)
    x[np.arange(n), t] += 1.0
    return x, t


def success(t, rng):
    """(n, K) success matrix. On slice k only worker k is reliable."""
    n = len(t)
    p = np.full((n, N_WORKERS), 0.05, dtype=np.float32)
    gen = t == 0
    p[gen, 0] = 0.90
    p[gen, 1:] = 0.10
    spec = ~gen
    p[spec, 0] = 0.35
    p[spec[np.newaxis, :][0], t[spec]] = 0.95
    return (rng.random((n, N_WORKERS)) < p).astype(np.float32)


def run(condition: str = "shift", seed: int = 42, out_dir: Path = DATA) -> Path:
    assert condition in CONDITIONS
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    strength = {"shift": 0.05, "mild_shift": 0.5,
                "entropy_reg": 0.05, "stable": 1.0}[condition]
    # entropy_reg raises exploration at the shift -- the mitigation you would
    # actually deploy once you suspected the traffic change was a risk.
    beta_after = BETA_NULL if condition == "entropy_reg" else BETA_BASE

    policy = Router()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    # Held-out evaluation set, fixed for the whole run and never trained on.
    ev_x, ev_t = sample_tasks(N_EVAL, np.random.default_rng(seed + 777),
                              slice_probs(False, 1.0))
    ev_xt = torch.from_numpy(ev_x)
    ev_masks = [ev_t == k for k in range(1, N_WORKERS)]

    log = {k: [] for k in ("pi_mean", "g", "grad_norm", "entropy",
                           "reward", "coverage", "eval_step")}

    for step in range(STEPS):
        p_type = slice_probs(step >= T_SHIFT, strength)
        x_np, t = sample_tasks(BATCH, rng, p_type)
        succ = success(t, rng)

        probs = F.softmax(policy(torch.from_numpy(x_np)), dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a_t = dist.sample()
        action = a_t.numpy()

        rewards = torch.from_numpy(succ[np.arange(BATCH), action])
        advantage = rewards - rewards.mean()
        ent = dist.entropy().mean()
        beta = beta_after if step >= T_SHIFT else BETA_BASE
        loss = -(advantage * dist.log_prob(a_t)).mean() - beta * ent

        opt.zero_grad()
        loss.backward()

        # Per-worker pressure, read before the update is applied.
        p_np = probs.detach().numpy()
        log["g"].append(pressure(advantage.numpy(), action, p_np,
                                 per_worker=True))
        log["pi_mean"].append(p_np.mean(axis=0))
        log["grad_norm"].append(float(sum(
            p.grad.pow(2).sum() for p in policy.parameters()
            if p.grad is not None) ** 0.5))
        log["entropy"].append(float(ent.detach()))
        log["reward"].append(float(rewards.mean()))

        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            with torch.no_grad():
                choice = policy(ev_xt).argmax(dim=-1).numpy()
            # c_k = P(route to k | held-out task from slice k)
            log["coverage"].append(
                np.array([float((choice[m] == k + 1).mean())
                          for k, m in enumerate(ev_masks)])
            )
            log["eval_step"].append(step)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"partial_{condition}_seed{seed}.npz"
    np.savez_compressed(path, **{k: np.array(v) for k, v in log.items()},
                        t_shift=T_SHIFT, rare=np.array(RARE), seed=seed)
    return path


def failure_time(npz):
    """Failure step for one saved run, or None. Thin wrapper over the method."""
    ev = npz["eval_step"]
    i0 = int(np.argmax(ev >= npz["t_shift"]))
    return coverage_failure(npz["coverage"], ev, start_idx=i0,
                            dead_thr=DEAD_THR, n_dead=N_DEAD_FAIL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    a = p.parse_args()
    for c in a.conditions:
        for s in a.seeds:
            path = run(c, s)
            t, alive = failure_time(np.load(path))
            print(f"{c:12s} seed {s:2d}: alive at shift {alive.sum():2d}/19  "
                  f"failure {t}")
