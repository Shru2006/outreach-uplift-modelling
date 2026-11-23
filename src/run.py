"""Train every learner, evaluate on the held-out experiment, write reports.

Run:  python -m src.run
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (BUDGET_TABLE_PATH, BUDGETS, FEATURES, FIGURES_DIR, METRICS_PATH,
                        RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE, TREATMENT, ensure_dirs)
from src.evaluate import (observed_uplift_in_top, qini_coefficient, qini_curve, segment,
                          true_incremental_per_1000)
from src.learners import all_learners, fit_predict

# Reference categorical palette (fixed order) + neutral inks.
COLORS = {
    "Propensity (rank by P(apply))": "#eb6834",
    "S-learner": "#eda100",
    "T-learner": "#1baf7a",
    "X-learner": "#2a78d6",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(RAW_DATA_PATH)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df[[TREATMENT, TARGET]])
    X_tr, X_te = train[FEATURES], test[FEATURES]
    t_te, y_te, u_te = test[TREATMENT].values, test[TARGET].values, test["true_uplift"].values

    scores = {}
    for learner in all_learners():
        scores[learner.name] = fit_predict(learner, X_tr, train[TREATMENT], train[TARGET], X_te)
        print(f"fitted {learner.name}")

    rng = np.random.default_rng(RANDOM_STATE)
    reference = {"Random targeting": rng.random(len(test)), "Oracle (true uplift)": u_te}

    rows, budget_rows = [], []
    for name, s in {**scores, **reference}.items():
        row = {
            "model": name,
            "qini_coefficient": round(qini_coefficient(s, t_te, y_te), 4),
            "spearman_vs_true_uplift": round(pd.Series(s).corr(pd.Series(u_te), method="spearman"), 4),
            "observed_uplift_top20": round(observed_uplift_in_top(s, t_te, y_te, 0.20), 4),
        }
        for b in BUDGETS:
            val = round(true_incremental_per_1000(s, u_te, b), 1)
            row[f"incremental_per_1000_calls_top{int(b * 100)}"] = val
            budget_rows.append({"model": name, "budget_share": b, "incremental_per_1000_calls": val})
        rows.append(row)
    results = pd.DataFrame(rows)
    pd.DataFrame(budget_rows).to_csv(BUDGET_TABLE_PATH, index=False)

    # Segment mix of the top 20% chosen by propensity vs the best uplift model.
    seg = segment(test["p_apply_if_not_contacted"].values, u_te)
    learner_rows = results[results.model.isin(COLORS) & (results.model != "Propensity (rank by P(apply))")]
    best = learner_rows.sort_values("qini_coefficient", ascending=False).model.iloc[0]
    k = int(len(test) * 0.20)
    mix = {}
    for name in ["Propensity (rank by P(apply))", best]:
        top = np.argsort(-scores[name], kind="stable")[:k]
        mix[name] = pd.Series(seg[top]).value_counts(normalize=True).round(4).to_dict()
    population_mix = pd.Series(seg).value_counts(normalize=True).round(4).to_dict()

    metrics = {
        "dataset": {
            "customers": int(len(df)),
            "test_customers": int(len(test)),
            "apply_rate_contacted": round(float(df.loc[df[TREATMENT] == 1, TARGET].mean()), 4),
            "apply_rate_not_contacted": round(float(df.loc[df[TREATMENT] == 0, TARGET].mean()), 4),
            "average_true_uplift": round(float(df["true_uplift"].mean()), 4),
        },
        "best_uplift_model": best,
        "models": results.to_dict(orient="records"),
        "segment_mix_top20": mix,
        "segment_mix_population": population_mix,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(results.to_string(index=False))
    print(json.dumps(mix, indent=2))

    _plot_qini(scores, t_te, y_te, u_te)
    _plot_budget(results, best)
    _plot_segments(mix, best, population_mix)


def _plot_qini(scores, t, y, u):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    _style(ax)
    for name, s in scores.items():
        c = qini_curve(s, t, y)
        ax.plot(c.share_targeted * 100, c.incremental, color=COLORS[name], linewidth=2,
                label=name.split(" (")[0])
    total = qini_curve(u, t, y).incremental.iloc[-1]
    ax.plot([0, 100], [0, total], color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("Random targeting", (70, total * 0.7), xytext=(4, -12), textcoords="offset points",
                fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("Share of customers called (%)", color=INK, fontsize=10)
    ax.set_ylabel("Incremental applications (test set)", color=INK, fontsize=10)
    ax.set_title("Qini curves: uplift models find the customers a call actually moves",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qini_curves.png")
    plt.close(fig)


def _plot_budget(results, best):
    names = ["Random targeting", "Propensity (rank by P(apply))", best, "Oracle (true uplift)"]
    colors = ["#c3c2b7", COLORS["Propensity (rank by P(apply))"], COLORS[best], "#52514e"]
    labels = ["Random", "Propensity", best, "Oracle (upper bound)"]
    budgets = [10, 20, 30]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _style(ax)
    width = 0.2
    x = np.arange(len(budgets))
    for i, (n, col, lab) in enumerate(zip(names, colors, labels)):
        vals = [results.loc[results.model == n, f"incremental_per_1000_calls_top{b}"].iloc[0] for b in budgets]
        bars = ax.bar(x + (i - 1.5) * width, vals, width - 0.02, color=col, label=lab)
        if n in (best, "Propensity (rank by P(apply))"):
            for bar, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, max(v, 0)),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [f"Call top {b}%" for b in budgets])
    ax.set_ylabel("Extra applications per 1,000 calls", color=INK, fontsize=10)
    ax.set_title("Same call budget, more incremental applications", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "incremental_by_budget.png")
    plt.close(fig)


def _plot_segments(mix, best, population):
    order = ["Persuadable", "Sure thing", "Lost cause", "Sleeping dog"]
    seg_colors = ["#2a78d6", "#1baf7a", "#c3c2b7", "#e34948"]
    groups = [("Whole base", population), ("Top 20% by propensity", mix["Propensity (rank by P(apply))"]),
              (f"Top 20% by {best}", mix[best])]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
    for i, (label, m) in enumerate(groups):
        left = 0
        for s, col in zip(order, seg_colors):
            v = m.get(s, 0) * 100
            ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.6,
                    label=s if i == 0 else None)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if col != "#c3c2b7" else INK)
            left += v
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of customers (%)", color=INK, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Who gets called: uplift targeting skips sure things and sleeping dogs",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "segment_mix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
