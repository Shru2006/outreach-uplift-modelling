"""Synthetic randomised outreach experiment.

Every customer has two potential outcomes: the chance they apply for a home loan
if a lender calls them, and the chance if nobody calls. A coin flip decides who
is called, exactly like a real A/B test. Because the data is synthetic we also
keep both probabilities, so models can be scored against the true uplift, which
is never observable in real life.

The hidden process creates the four classic uplift segments:
  * Persuadables   - deposit-ready, still undecided: a call helps a lot
  * Sure things    - already deep in the buying journey: they apply anyway
  * Lost causes    - not ready: a call changes nothing
  * Sleeping dogs  - over-contacted or recently declined: a call puts them off

Run:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, RAW_DATA_PATH, ensure_dirs


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate(n: int = 80_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 50, n)
    annual_income = np.round(rng.lognormal(np.log(92_000), 0.35, n).clip(40_000, 400_000), -2)
    deposit_ratio = (rng.beta(2, 5, n) * 0.7).round(3)           # savings / income
    savings_balance = np.round(annual_income * deposit_ratio, -1)
    rent_to_income = rng.normal(0.28, 0.07, n).clip(0.08, 0.6).round(3)
    digital_intent = rng.beta(1.6, 3.5, n).round(3)               # 0-1 research activity
    savings_growth_6m = rng.normal(0.05, 0.12, n).round(3)
    bnpl_txn_3m = rng.poisson(1.5, n)
    tenure_years = rng.gamma(2, 2.5, n).clip(0.2, 25).round(1)
    recent_credit_decline = rng.binomial(1, 0.07, n)
    marketing_contacts_90d = rng.poisson(1.1, n)

    # Probability of applying if nobody calls (log-odds scale).
    base = (
        -5.1
        + 4.2 * digital_intent
        + 3.0 * deposit_ratio
        + 1.5 * savings_growth_6m
        - 0.12 * bnpl_txn_3m
        + 0.03 * np.minimum(tenure_years, 10)
        - 0.8 * recent_credit_decline
        + 0.004 * (annual_income / 1000 - 92)
        + 1.2 * (digital_intent > 0.6)
    )

    # What a call changes (log-odds scale). Strongest for deposit-ready customers
    # whose research is moderate: they are close but undecided.
    deposit_ready = _sigmoid((deposit_ratio - 0.15) * 25)
    undecided = np.exp(-(((digital_intent - 0.35) / 0.17) ** 2))
    effect = (
        1.2 * deposit_ready * undecided
        + 0.4 * deposit_ready * (savings_growth_6m > 0.08)
        - 0.45 * np.maximum(marketing_contacts_90d - 1, 0)
        - 1.1 * recent_credit_decline
        + 0.05
    )

    p0 = _sigmoid(base)
    p1 = _sigmoid(base + effect)

    contacted = rng.binomial(1, 0.5, n)
    applied = rng.binomial(1, np.where(contacted == 1, p1, p0))

    return pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "age": age,
        "annual_income": annual_income,
        "savings_balance": savings_balance,
        "deposit_ratio": deposit_ratio,
        "rent_to_income": rent_to_income,
        "digital_intent": digital_intent,
        "savings_growth_6m": savings_growth_6m,
        "bnpl_txn_3m": bnpl_txn_3m,
        "tenure_years": tenure_years,
        "recent_credit_decline": recent_credit_decline,
        "marketing_contacts_90d": marketing_contacts_90d,
        "contacted": contacted,
        "applied_90d": applied,
        "p_apply_if_not_contacted": p0.round(5),
        "p_apply_if_contacted": p1.round(5),
        "true_uplift": (p1 - p0).round(5),
    })


def main() -> None:
    ensure_dirs()
    df = generate()
    df.to_csv(RAW_DATA_PATH, index=False)
    t, c = df[df.contacted == 1], df[df.contacted == 0]
    print(f"Wrote {len(df):,} customers to {RAW_DATA_PATH}")
    print(f"Apply rate if contacted: {t.applied_90d.mean():.2%} | not contacted: {c.applied_90d.mean():.2%}")
    print(f"Average true uplift: {df.true_uplift.mean():.2%}")


if __name__ == "__main__":
    main()
