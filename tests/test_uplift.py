"""Fast checks on a small synthetic sample (no saved files needed).

Run:  pytest -q
"""
import numpy as np
import pytest

from src.config import FEATURES, PROTECTED_COLS, TARGET, TREATMENT, TRUTH_COLS
from src.evaluate import qini_coefficient, qini_curve, segment, true_incremental_per_1000
from src.generate_data import generate
from src.learners import SLearner, XLearner, fit_predict


@pytest.fixture(scope="module")
def sample():
    return generate(n=20_000, seed=7)


def test_treatment_is_randomised_and_balanced(sample):
    assert 0.48 < sample[TREATMENT].mean() < 0.52
    # Randomisation means features look the same in both groups.
    g = sample.groupby(TREATMENT)["digital_intent"].mean()
    assert abs(g[1] - g[0]) < 0.01


def test_calling_helps_on_average_but_not_everyone(sample):
    assert sample["true_uplift"].mean() > 0
    assert (sample["true_uplift"] < 0).any(), "sleeping dogs must exist"


def test_features_exclude_outcome_truth_and_protected(sample):
    forbidden = {TARGET, TREATMENT, *TRUTH_COLS, *PROTECTED_COLS}
    assert forbidden.isdisjoint(FEATURES)


def test_oracle_beats_random_on_qini(sample):
    rng = np.random.default_rng(0)
    t, y = sample[TREATMENT].values, sample[TARGET].values
    assert qini_coefficient(sample["true_uplift"], t, y) > qini_coefficient(rng.random(len(sample)), t, y)


def test_qini_curve_ends_at_total_incremental(sample):
    c = qini_curve(sample["true_uplift"], sample[TREATMENT], sample[TARGET])
    assert c["share_targeted"].iloc[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("learner", [SLearner(), XLearner()])
def test_learners_rank_true_uplift(sample, learner):
    train, test = sample.iloc[:14_000], sample.iloc[14_000:]
    pred = fit_predict(learner, train[FEATURES], train[TREATMENT], train[TARGET], test[FEATURES])
    corr = np.corrcoef(pred, test["true_uplift"])[0, 1]
    assert corr > 0.3
    assert true_incremental_per_1000(pred, test["true_uplift"], 0.2) > test["true_uplift"].mean() * 1000


def test_segments_cover_everyone(sample):
    seg = segment(sample["p_apply_if_not_contacted"], sample["true_uplift"])
    assert set(seg) <= {"Persuadable", "Sure thing", "Lost cause", "Sleeping dog"}
