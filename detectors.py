"""Early warning for partial routing collapse.

An orchestrator routes tasks to workers. Partial collapse is when a few workers
stop being routed to while the aggregate routing distribution still looks
healthy -- three specialists out of twenty going dark barely moves `max_k pi_k`.

This module is the method, and it is deliberately small:

    coverage_failure   define the failure by held-out behaviour, not by a
                       threshold on anything a monitor reads
    pressure           the candidate early-warning signal, per worker
    sweep              compare monitors at a matched false-alarm rate
    table_at_far       read off the best operating point within a FAR budget

Everything takes numpy arrays. Nothing imports torch. To use this on a real
system you log the arrays described in each docstring and call these functions.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def ema(x, alpha: float = 0.08) -> np.ndarray:
    """Exponential moving average. Per-step monitors are far too noisy raw."""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def first_persistent(mask, persist: int = 20, start: int = 0):
    """First index where `mask` held for `persist` consecutive steps.

    Persistence is what separates an alarm from single-step noise. Returns
    None if it never happens.
    """
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= persist:
            return i - persist + 1
    return None


# ---------------------------------------------------------------------------
# defining the failure
# ---------------------------------------------------------------------------

def coverage_failure(coverage, eval_steps, start_idx: int = 0,
                     dead_thr: float = 0.5, n_dead: int = 3,
                     persist: int = 3):
    """When did the system lose `n_dead` workers, judged on held-out behaviour?

    coverage : (T_eval, K) where coverage[t, k] = P(route to worker k | held-out
        task that worker k is the right choice for). Measured on a fixed
        evaluation set, not on the training stream.
    eval_steps : (T_eval,) training step each row was measured at.
    start_idx : index of the first row to consider (e.g. the onset of the
        change you are studying).

    A worker counts as dead when its coverage drops below `dead_thr`. Only
    workers that were alive at `start_idx` can die, so a worker that never
    learned its slice cannot manufacture a failure.

    Returns (failure_step, alive_at_start). failure_step is None if the run
    never fails -- that run is a NULL.

    Why this and not a threshold on the routing distribution: a monitor scored
    against a definition written in its own units wins by construction, and the
    lead it appears to have is just the distance between two thresholds on one
    quantity. Define the failure by what you actually lose.
    """
    coverage = np.asarray(coverage, dtype=float)
    eval_steps = np.asarray(eval_steps)

    alive = coverage[start_idx] >= dead_thr
    if alive.sum() < n_dead:
        return None, alive

    dead = (coverage < dead_thr) & alive[None, :]
    run = 0
    for i in range(start_idx, len(eval_steps)):
        run = run + 1 if dead[i].sum() >= n_dead else 0
        if run >= persist:
            return int(eval_steps[i - persist + 1]), alive
    return None, alive


# ---------------------------------------------------------------------------
# the candidate signal
# ---------------------------------------------------------------------------

def pressure(advantage, action, pi, per_worker: bool = False):
    """Advantage-weighted gradient pressure toward each worker, one step.

    For a softmax policy,  d log pi(a) / d logit_k = 1[a=k] - pi_k,  so

        g_k = E[ A * (1[a=k] - pi_k) ]

    is the mean push toward worker k in the update about to be applied. Read it
    before the optimiser step -- that is what makes it a leading indicator.

    per_worker=True returns the (K,) vector; the alarm statistic for partial
    collapse is `min_k g_k`, the most negative pressure, which names the worker
    being squeezed out. per_worker=False returns `max_k g_k`, which is only
    useful for total collapse onto one worker and is blind to partial collapse.

    The signal is a TRANSIENT, not a level. As pi_k approaches 0 the worker
    stops being sampled, so g_k = E[A(0 - pi_k)] = -pi_k E[A] -> 0: the
    pressure vanishes once the worker is already gone. You are looking for a
    trough that precedes the death, not a level that persists after it.

    advantage : (B,) reward minus baseline
    action    : (B,) index of the sampled worker
    pi        : (B, K) action probabilities
    """
    A = np.asarray(advantage, dtype=float)
    a = np.asarray(action, dtype=int)
    p = np.asarray(pi, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(a)), a] = 1.0
    g = (A[:, None] * (onehot - p)).mean(axis=0)
    return g if per_worker else float(g.max())


# ---------------------------------------------------------------------------
# comparing monitors honestly
# ---------------------------------------------------------------------------

def sweep(positives, nulls, n_thr: int = 160, start: int = 0,
          persist: int = 20) -> dict:
    """Warning time versus false-alarm rate, for one monitor.

    positives : list of (stat, t_failure) for runs that failed.
    nulls     : list of stat arrays for runs that did not.
        A null has to be a run where the signal moves and the failure still
        does not arrive. A run where nothing happens hands every monitor a
        perfect false-alarm rate for free and measures nothing.
    start : ignore everything before this step.

    Returns {'points': [{thr, far, dr, mean_lead, min_lead}, ...]} where
      far  -- fraction of null runs the monitor alarms on
      dr   -- fraction of failures alarmed before they happened
      lead -- steps of warning, averaged over the failures it caught

    Never compare two monitors at different false-alarm rates: one that alarms
    constantly wins on lead and is worthless.
    """
    pooled = np.concatenate([s[start:] for s in nulls]
                            + [s[start:] for s, _ in positives])
    grid = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, n_thr)))

    points = []
    for thr in grid:
        n_fa = sum(first_persistent(s > thr, persist, start) is not None
                   for s in nulls)
        leads = [t - a for s, t in positives
                 if (a := first_persistent(s > thr, persist, start)) is not None
                 and a < t]
        points.append({
            "thr": float(thr),
            "far": n_fa / len(nulls),
            "dr": len(leads) / len(positives),
            "mean_lead": float(np.mean(leads)) if leads else None,
            "min_lead": float(np.min(leads)) if leads else None,
        })
    return {"points": points, "n_positive": len(positives), "n_null": len(nulls)}


def table_at_far(curves: dict, far_budget: float) -> dict:
    """Best operating point subject to far <= far_budget.

    Highest detection rate, ties broken by lead. `feasible=False` means the
    monitor cannot reach that false-alarm rate while detecting anything.
    """
    ok = [p for p in curves["points"]
          if p["far"] <= far_budget + 1e-9 and p["dr"] > 0]
    if not ok:
        return {"feasible": False, "far": None, "dr": 0.0,
                "mean_lead": None, "min_lead": None, "thr": None}
    return {"feasible": True, **max(ok, key=lambda p: (p["dr"],
                                                       p["mean_lead"] or 0.0))}
