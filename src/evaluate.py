"""Uplift evaluation.

Two views:
  * Observed (what you could do with a real experiment): Qini curves and the
    contacted-minus-not-contacted application rate inside each targeted slice.
  * Oracle (only possible on synthetic data): the true incremental
    applications each targeting rule would buy, from the known uplift.
"""
import numpy as np
import pandas as pd


def qini_curve(score, treatment, y, n_points: int = 100) -> pd.DataFrame:
    """Incremental applications as a growing share of customers is targeted.

    At each cut-off k: Y_t(k) - Y_c(k) * N_t(k) / N_c(k), where t and c are the
    contacted and not-contacted customers among the top k by score.
    """
    order = np.argsort(-np.asarray(score), kind="stable")
    t = np.asarray(treatment)[order]
    y = np.asarray(y)[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    return pd.DataFrame({"share_targeted": (idx + 1) / len(y), "incremental": qini[idx]})


def qini_coefficient(score, treatment, y) -> float:
    """Area between the model's Qini curve and the random-targeting line,
    as a share of the total incremental applications in the population."""
    curve = qini_curve(score, treatment, y, n_points=200)
    total = curve["incremental"].iloc[-1]
    random_line = curve["share_targeted"] * total
    area = np.trapezoid(curve["incremental"] - random_line, curve["share_targeted"])
    return float(area / abs(total)) if total else float("nan")


def observed_uplift_in_top(score, treatment, y, share: float) -> float:
    """Application rate (contacted) minus (not contacted) within the top `share`."""
    k = int(len(score) * share)
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    t = np.asarray(treatment)[top]
    yy = np.asarray(y)[top]
    return float(yy[t == 1].mean() - yy[t == 0].mean())


def true_incremental_per_1000(score, true_uplift, share: float) -> float:
    """Extra applications per 1,000 calls if the top `share` were called."""
    k = int(len(score) * share)
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(np.asarray(true_uplift)[top].mean() * 1000)


def segment(p0, true_uplift) -> np.ndarray:
    """Label each customer with a classic uplift segment using the true values."""
    p0 = np.asarray(p0)
    u = np.asarray(true_uplift)
    return np.select(
        [u < -0.005, u >= 0.03, p0 >= 0.20],
        ["Sleeping dog", "Persuadable", "Sure thing"],
        default="Lost cause",
    )
