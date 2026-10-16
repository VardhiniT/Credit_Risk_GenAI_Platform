"""WOE + logistic regression scorecard.

    PD     = probability of the target event (e.g. 3+ months delinquent)
    Score  = offset + factor * ln( (1 - PD) / PD )          higher score = safer
    factor = pdo / ln(2),  offset = base_score - factor * ln(base_odds)

Score and PD are two views of the same number, so they can never disagree. The points table
(points per bin) sums to exactly the same score and is what powers reason codes.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .woe import WOEBinner


class Scorecard:
    def __init__(self, cfg):
        self.cfg = cfg
        w, s = cfg["woe"], cfg["scorecard"]
        self.binner = WOEBinner(w["max_bins"], w["min_bin_pct"], w["monotonic_numeric"])
        self.factor = s["pdo"] / np.log(2)
        self.offset = s["base_score"] - self.factor * np.log(s["base_odds"])
        self.selected_, self.dropped_ = [], {}
        self.model_ = None
        self.band_cutoffs_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, df, y):
        w, f = self.cfg["woe"], self.cfg["features"]
        flags = list(f["derived"])
        numeric = list(f["numeric"])
        cat = list(f["categorical"]) + flags            # 0/1 flags are binned as categories
        df = self._prep(df)                              # flags -> strings (binned as categories)
        self.binner.fit(df, y, numeric, cat)

        iv = self.binner.iv_
        keep = [c for c in iv if iv[c] >= w["min_iv"]]
        for c in iv:
            if c not in keep:
                self.dropped_[c] = f"IV {iv[c]:.4f} below {w['min_iv']}"
        keep = self._drop_correlated(df, keep, w["max_corr"])
        self.selected_ = self._fit_with_sign_check(df, y, keep)
        self._set_band_cutoffs(self.score(df, already_flagged=True))
        return self

    def _drop_correlated(self, df, keep, max_corr):
        W = self.binner.transform(df, keep)
        order = sorted(keep, key=lambda c: -self.binner.iv_[c])
        chosen = []
        for c in order:
            clash = [k for k in chosen if abs(np.corrcoef(W[c], W[k])[0, 1]) > max_corr]
            if clash:
                self.dropped_[c] = f"WOE correlated > {max_corr} with {clash[0]} (higher IV kept)"
            else:
                chosen.append(c)
        return chosen

    def _fit_with_sign_check(self, df, y, keep):
        """WOE = ln(good/bad) and y = bad, so every coefficient must be NEGATIVE."""
        keep = list(keep)
        while True:
            W = self.binner.transform(df, keep)
            m = LogisticRegression(C=1e6, max_iter=2000).fit(W, y)
            wrong = [(c, b) for c, b in zip(keep, m.coef_[0]) if b >= 0]
            if not wrong:
                self.model_ = m
                return keep
            c, b = max(wrong, key=lambda t: t[1])
            self.dropped_[c] = f"wrong-sign coefficient ({b:+.3f}) after fitting"
            keep.remove(c)

    # -------------------------------------------------------------- predict
    def _prep(self, df, already_flagged=False):
        df = df.copy()
        if not already_flagged:
            for c in self.cfg["features"]["derived"]:
                df[c] = df[c].astype(str)
        return df

    def predict_pd(self, df, already_flagged=False):
        W = self.binner.transform(self._prep(df, already_flagged), self.selected_)
        return self.model_.predict_proba(W)[:, 1]

    def score(self, df, already_flagged=False):
        p = np.clip(self.predict_pd(df, already_flagged), 1e-9, 1 - 1e-9)
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

    # ----------------------------------------------------- points & reasons
    def points_table(self):
        n = len(self.selected_)
        a = self.model_.intercept_[0]
        rows = []
        for c, beta in zip(self.selected_, self.model_.coef_[0]):
            t = self.binner.tables_[c]
            pts = -self.factor * beta * t["woe"].to_numpy() + (self.offset - self.factor * a) / n
            rows.append(pd.DataFrame({"feature": c, "bin": t["bin"], "label": t["label"],
                                      "woe": t["woe"], "bad_rate": t["bad_rate"],
                                      "pct": t["pct"], "points": pts}))
        return pd.concat(rows, ignore_index=True)

    def reason_codes(self, df, top_n=3, already_flagged=False):
        """Points lost versus the best possible bin, per feature. Biggest losses = reasons."""
        df = self._prep(df, already_flagged)
        pt = self.points_table()
        best = pt.groupby("feature")["points"].max()
        loss = {}
        for c in self.selected_:
            p = pt[pt.feature == c].sort_values("bin")["points"].to_numpy()
            idx = self.binner.bin_index(df, c)
            loss[c] = best[c] - p[np.clip(idx, 0, len(p) - 1)]
        L = pd.DataFrame(loss, index=df.index)
        out = []
        for _, r in L.iterrows():
            top = r.sort_values(ascending=False).head(top_n)
            out.append([f"{k} (-{v:.1f} pts)" for k, v in top.items() if v > 0])
        return out

    # --------------------------------------------------------------- persist
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def coef_table(self):
        t = pd.DataFrame({"feature": self.selected_, "coef": self.model_.coef_[0],
                          "iv": [self.binner.iv_[c] for c in self.selected_]})
        t.loc[len(t)] = ["(intercept)", float(self.model_.intercept_[0]), np.nan]
        return t
