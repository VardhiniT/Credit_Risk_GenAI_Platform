"""Scorecard built on the scorecardpy library (second engine, selectable in train.py).

Same steps as our own scorecard.py, but scorecardpy does the binning and WOE:

  1. sc.woebin       cut every feature into bins and compute WOE and IV (tree-based binning)
  2. keep features   IV >= min_iv, then drop one of every highly correlated pair of WOE columns
  3. logistic regression on the WOE columns, repeated after dropping any wrong-sign feature
  4. sc.scorecard    points per bin (used for the points table)
  5. score = offset + factor * ln((1 - PD) / PD)   exact, so score and PD can never disagree

SIGN CONVENTION (differs from our own woe.py):
    scorecardpy:  WOE = ln( share of BAD in the bin / share of GOOD in the bin )
    -> POSITIVE WOE = riskier, and every regression coefficient must be POSITIVE.

scorecardpy has no switch to force a monotone trend, so woe.monotonic_numeric from the
config is not applied in this engine.
"""
import warnings
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import model_inputs

TARGET = "_target_"


@contextmanager
def _quiet():
    """scorecardpy is an older library and prints hundreds of pandas FutureWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _sc():
    import scorecardpy as sc    # imported here so the rest of the project works without it
    return sc


class SPScorecard:
    method_text = (
        "scorecardpy tree-based WOE binning (max bins and minimum bin share from the config), "
        "IV filter, correlation filter, unpenalised logistic regression, sign check "
        "(under scorecardpy's convention every coefficient must be positive). "
        "Monotone trend is not forced in this engine."
    )

    def __init__(self, cfg):
        self.cfg = cfg
        s = cfg["scorecard"]
        self.factor = s["pdo"] / np.log(2)
        self.offset = s["base_score"] - self.factor * np.log(s["base_odds"])
        self.bins_, self.iv_ = {}, {}
        self.selected_, self.dropped_ = [], {}
        self.model_ = None
        self.card_ = None
        self.band_cutoffs_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, df, y):
        w = self.cfg["woe"]
        sc = _sc()
        cols = [c for c in model_inputs(self.cfg) if c in df.columns]
        data = df[cols].reset_index(drop=True).copy()
        y = np.asarray(y).astype(int)
        data[TARGET] = y

        with _quiet():
            self.bins_ = sc.woebin(data, y=TARGET, x=cols, bin_num_limit=w["max_bins"],
                                   count_distr_limit=w["min_bin_pct"], method="tree",
                                   no_cores=1, print_step=0)
        self.iv_ = {v: float(b["total_iv"].iloc[0]) for v, b in self.bins_.items()}

        for c in cols:
            if c not in self.bins_:
                self.dropped_[c] = "constant column, no bins"
        ranked = sorted(self.iv_, key=lambda c: -self.iv_[c])
        keep = [c for c in ranked if self.iv_[c] >= w["min_iv"]]
        for c in ranked:
            if c not in keep:
                self.dropped_[c] = f"IV {self.iv_[c]:.4f} below {w['min_iv']}"
        keep = self._drop_correlated(data, keep, w["max_corr"])
        self.selected_ = self._fit_with_sign_check(data, y, keep)

        s = self.cfg["scorecard"]
        with _quiet():
            self.card_ = sc.scorecard({c: self.bins_[c] for c in self.selected_}, self.model_,
                                      [f"{c}_woe" for c in self.selected_],
                                      points0=s["base_score"], odds0=1 / s["base_odds"], pdo=s["pdo"])
        self._set_band_cutoffs(self.score(data))
        return self

    def _woe(self, df, cols):
        """Replace each value by its bin's WOE. Unseen category or unseen missing -> 0 (neutral)."""
        sc = _sc()
        d = df[cols].reset_index(drop=True)
        with _quiet():
            out = sc.woebin_ply(d, {c: self.bins_[c] for c in cols}, no_cores=1, print_step=0)
        W = out[[f"{c}_woe" for c in cols]].copy()
        W.columns = list(cols)
        return W.fillna(0.0)

    def _drop_correlated(self, data, keep, max_corr):
        if len(keep) < 2:
            return keep
        W = self._woe(data, keep)
        chosen = []
        for c in keep:                                  # already ordered by IV, strongest first
            clash = [k for k in chosen if abs(np.corrcoef(W[c], W[k])[0, 1]) > max_corr]
            if clash:
                self.dropped_[c] = f"WOE correlated > {max_corr} with {clash[0]} (higher IV kept)"
            else:
                chosen.append(c)
        return chosen

    def _fit_with_sign_check(self, data, y, keep):
        """WOE = ln(bad/good) and y = bad, so every coefficient must be POSITIVE."""
        keep = list(keep)
        while True:
            W = self._woe(data, keep)
            m = LogisticRegression(C=1e6, max_iter=2000).fit(W, y)
            wrong = [(c, b) for c, b in zip(keep, m.coef_[0]) if b <= 0]
            if not wrong or len(keep) == 1:
                self.model_ = m
                return keep
            c, b = min(wrong, key=lambda t: t[1])
            self.dropped_[c] = f"wrong-sign coefficient ({b:+.3f}) after fitting"
            keep.remove(c)

    # -------------------------------------------------------------- predict
    def predict_pd(self, df, already_flagged=False):
        return self.model_.predict_proba(self._woe(df, self.selected_))[:, 1]

    def score(self, df, already_flagged=False):
        p = np.clip(self.predict_pd(df), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - p) / p)

    # -------------------------------------------------------- risk bands
    def _set_band_cutoffs(self, train_scores):
        ups = [b["upper_pct"] for b in self.cfg["risk_bands"]][:-1]
        self.band_cutoffs_ = [float(x) for x in np.percentile(train_scores, ups)]

    @property
    def band_names(self):
        return [b["name"] for b in self.cfg["risk_bands"]]

    def band(self, scores):
        idx = np.searchsorted(np.array(self.band_cutoffs_), scores, side="left")
        return np.array(self.band_names)[idx]

    # ------------------------------------------------------------- tables
    def iv_table(self):
        return (pd.Series(self.iv_, name="iv").sort_values(ascending=False)
                .rename_axis("feature").reset_index())

    def bins_table(self):
        rows = []
        for c, b in self.bins_.items():
            b = b.reset_index(drop=True)
            rows.append(pd.DataFrame({
                "feature": c, "bin": range(len(b)), "label": b["bin"].astype(str).to_numpy(),
                "count": b["count"].to_numpy(), "pct": b["count_distr"].to_numpy(),
                "good": b["good"].to_numpy(), "bad": b["bad"].to_numpy(),
                "bad_rate": b["badprob"].to_numpy(), "woe": b["woe"].to_numpy(),
                "iv_contrib": b["bin_iv"].to_numpy()}))
        return pd.concat(rows, ignore_index=True)

    def coef_table(self):
        t = pd.DataFrame({"feature": self.selected_, "coef": self.model_.coef_[0],
                          "iv": [self.iv_[c] for c in self.selected_]})
        t.loc[len(t)] = ["(intercept)", float(self.model_.intercept_[0]), np.nan]
        return t

    def points_table(self):
        rows = []
        for c in self.selected_:
            b = self.bins_[c].reset_index(drop=True).assign(label=lambda d: d["bin"].astype(str))
            card = self.card_[c][["bin", "points"]].assign(label=lambda d: d["bin"].astype(str))
            m = b.merge(card[["label", "points"]], on="label", how="left")
            rows.append(pd.DataFrame({
                "feature": c, "bin": range(len(m)), "label": m["label"], "woe": m["woe"],
                "bad_rate": m["badprob"], "pct": m["count_distr"], "points": m["points"]}))
        base = float(self.card_["basepoints"]["points"].iloc[0])
        rows.append(pd.DataFrame({"feature": ["(base points)"], "bin": [0], "label": [""],
                                  "woe": [np.nan], "bad_rate": [np.nan], "pct": [np.nan],
                                  "points": [base]}))
        return pd.concat(rows, ignore_index=True)

    def reason_codes(self, df, top_n=3, already_flagged=False):
        """Points lost versus the safest bin of each feature. Biggest losses = reasons.
        Under scorecardpy's convention a higher WOE is riskier, and points = -factor*coef*WOE + const."""
        W = self._woe(df, self.selected_)
        coef = dict(zip(self.selected_, self.model_.coef_[0]))
        loss = pd.DataFrame({c: (self.factor * coef[c] * (W[c] - self.bins_[c]["woe"].min())).clip(lower=0)
                             for c in self.selected_})
        out = []
        for _, r in loss.iterrows():
            top = r.sort_values(ascending=False).head(top_n)
            out.append([f"{k} (-{v:.1f} pts)" for k, v in top.items() if v > 0])
        return out

    # --------------------------------------------------------------- persist
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
