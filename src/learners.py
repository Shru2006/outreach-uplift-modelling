"""Uplift learners built on LightGBM, written out by hand so each step is visible.

Every learner exposes .fit(X, treatment, y) and .predict_uplift(X), where uplift
is P(apply | contacted) - P(apply | not contacted) for each customer.

* PropensityBaseline - the usual approach: rank by P(apply). Not an uplift model;
  included because it is what most targeting does today.
* SLearner  - one model with the treatment flag as a feature.
* TLearner  - separate models for contacted and not-contacted customers.
* XLearner  - T-learner first, then models the imputed individual effects
  (Kunzel et al., 2019). Usually best when effects are small and noisy.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import RANDOM_STATE

_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
)


def _clf():
    return LGBMClassifier(**_PARAMS)


def _reg():
    return LGBMRegressor(**_PARAMS)


class PropensityBaseline:
    name = "Propensity (rank by P(apply))"

    def fit(self, X, treatment, y):
        self.model = _clf().fit(X, y)
        return self

    def predict_uplift(self, X):
        # Not an uplift estimate: this is the score a propensity model ranks by.
        return self.model.predict_proba(X)[:, 1]


class SLearner:
    name = "S-learner"

    def fit(self, X, treatment, y):
        self.model = _clf().fit(X.assign(_t=treatment), y)
        return self

    def predict_uplift(self, X):
        p1 = self.model.predict_proba(X.assign(_t=1))[:, 1]
        p0 = self.model.predict_proba(X.assign(_t=0))[:, 1]
        return p1 - p0


class TLearner:
    name = "T-learner"

    def fit(self, X, treatment, y):
        t = np.asarray(treatment) == 1
        self.m1 = _clf().fit(X[t], y[t])
        self.m0 = _clf().fit(X[~t], y[~t])
        return self

    def predict_uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    name = "X-learner"

    def __init__(self, propensity: float = 0.5):
        # Contact was randomised 50/50, so the propensity score is known exactly.
        self.e = propensity

    def fit(self, X, treatment, y):
        t = np.asarray(treatment) == 1
        y = np.asarray(y)
        m1 = _clf().fit(X[t], y[t])
        m0 = _clf().fit(X[~t], y[~t])
        # Imputed individual effects.
        d1 = y[t] - m0.predict_proba(X[t])[:, 1]
        d0 = m1.predict_proba(X[~t])[:, 1] - y[~t]
        self.tau1 = _reg().fit(X[t], d1)
        self.tau0 = _reg().fit(X[~t], d0)
        return self

    def predict_uplift(self, X):
        return self.e * self.tau0.predict(X) + (1 - self.e) * self.tau1.predict(X)


def all_learners():
    return [PropensityBaseline(), SLearner(), TLearner(), XLearner()]


def fit_predict(learner, X_train: pd.DataFrame, t_train, y_train, X_test: pd.DataFrame):
    return learner.fit(X_train, np.asarray(t_train), np.asarray(y_train)).predict_uplift(X_test)
