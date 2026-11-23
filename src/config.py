"""Central paths and settings so every script reads from one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "outreach_experiment_synthetic.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"
BUDGET_TABLE_PATH = REPORTS_DIR / "incremental_by_budget.csv"

ID_COL = "customer_id"
TREATMENT = "contacted"          # 1 = lender called the customer (randomised)
TARGET = "applied_90d"           # 1 = applied for a home loan within 90 days

# Ground truth that only exists because the data is synthetic. Used to score
# the models, never as an input.
TRUTH_COLS = ["p_apply_if_not_contacted", "p_apply_if_contacted", "true_uplift"]

# Kept for reporting only, never a model input.
PROTECTED_COLS = ["age"]

FEATURES = [
    "annual_income",
    "savings_balance",
    "deposit_ratio",
    "rent_to_income",
    "digital_intent",
    "savings_growth_6m",
    "bnpl_txn_3m",
    "tenure_years",
    "recent_credit_decline",
    "marketing_contacts_90d",
]

RANDOM_STATE = 42
TEST_SIZE = 0.4
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
