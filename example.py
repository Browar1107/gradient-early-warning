"""Minimal end-to-end use of the method. Run: python example.py

Pure numpy. Swap the synthetic arrays for arrays logged from your own system
and nothing else changes. `toy_partial_collapse.py` is the real version of this;
here everything is faked so the whole method fits on one screen.
"""
import numpy as np

from detectors import coverage_failure, ema, pressure, sweep, table_at_far

rng = np.random.default_rng(0)
K, T, B, ONSET, EVAL_EVERY = 8, 800, 64, 200, 10


def fake_run(fails: bool):
    """One training run.

    Returns the per-step per-worker pressure and the held-out coverage. When
    `fails`, workers 1 and 2 come under reward pressure at ONSET and their
    routing decays; otherwise nothing happens and the run is a null.
    """
    g_hist, cov, ev = [], [], []
    logit = np.zeros(K)
    for t in range(T):
        squeezed = fails and t >= ONSET
        if squeezed:
            logit[1:3] -= 0.004
            logit -= logit.mean()      # a mitigation keeps the policy spread

        pi = np.exp(logit) / np.exp(logit).sum()
        action = rng.choice(K, size=B, p=pi)
        advantage = rng.normal(0, 0.3, B)
        if squeezed:                   # the reward stops favouring 1 and 2
            advantage -= 0.4 * np.isin(action, [1, 2])

        g_hist.append(pressure(advantage, action, np.tile(pi, (B, 1)),
                               per_worker=True))
        if t % EVAL_EVERY == 0:
            # coverage: does the policy still pick worker k for k's own tasks?
            cov.append((pi[1:] / pi[1:].max()).clip(0, 1))
            ev.append(t)
    return np.array(g_hist), np.array(cov), np.array(ev)


positives, nulls = [], []
for fails in (True,) * 6 + (False,) * 6:
    g, cov, ev = fake_run(fails)
    i0 = int(np.argmax(ev >= ONSET))
    t_fail, _ = coverage_failure(cov, ev, start_idx=i0, n_dead=2)
    stat = -ema(g.min(axis=1))          # alarm statistic: most negative pressure
    if t_fail is None:
        nulls.append(stat)
    else:
        positives.append((stat, t_fail))

print(f"{len(positives)} runs failed, {len(nulls)} did not")
curves = sweep(positives, nulls, start=ONSET)
for budget in (0.0, 0.10, 0.20):
    r = table_at_far(curves, budget)
    if r["feasible"]:
        print(f"  false alarms <= {budget:.0%}: detected {r['dr']:.0%}, "
              f"mean warning {r['mean_lead']:.0f} steps")
    else:
        print(f"  false alarms <= {budget:.0%}: nothing detectable")
