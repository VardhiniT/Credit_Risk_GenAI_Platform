"""The calculations behind the assistant. Everything numeric comes from here, never from the LLM.

Each method returns a plain dict of FACTS. The LLM (or a template) only turns facts into sentences.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import data
from ..config import load_config, model_inputs
from ..monitor import first_breach, load_baseline, thresholds
from . import scenarios as sc_mod
from .governance import LABELS, MODEL_FIELDS, UNKNOWN_IS_ZERO, GovernanceError, NeedsInfo, validate_fields

REQUIRED = ["CreditScore", "DTI", "LTV", "OrigUPB", "OrigInterestRate"]
DEFAULTS = {"MIP": 0, "Units": 1, "FirstTimeHomebuyer": "X", "Occupancy": "X", "Channel": "X", "PPM": "X",
            "PropertyType": "X", "LoanPurpose": "X", "NumBorrowers": "X"}
FLAG_LABELS = ("cs_unknown", "dti_unknown", "ltv_gt80")


class ModelService:
    def __init__(self, sc, cfg, baseline=None, monitoring_dir="monitoring"):
        self.sc, self.cfg, self.baseline = sc, cfg, baseline
        self.monitoring_dir = Path(monitoring_dir)

    @classmethod
    def from_artifacts(cls, target="primary", monitoring_dir="monitoring", cfg=None):
        cfg = cfg or load_config()
        folder = Path(cfg["paths"]["artifacts"]) / cfg["target"][target]
        sc = joblib.load(folder / "scorecard.joblib")
        base = load_baseline(folder / "baseline.json") if (folder / "baseline.json").exists() else None
        return cls(sc, cfg, base, monitoring_dir)

    # ------------------------------------------------------------------ scoring
    def _check_required(self, loan, label="loan"):
        missing = [k for k in REQUIRED if k not in loan]
        if missing:
            raise NeedsInfo(f"The {label} is missing: {', '.join(LABELS.get(k, k) for k in missing)}. "
                            f"Please add them (other fields are optional).")

    def _prepared_many(self, loans):
        rows = [{**DEFAULTS, **{k: v for k, v in l.items() if k in MODEL_FIELDS}} for l in loans]
        return data.prepare(pd.DataFrame(rows), self.cfg, with_targets=False)[model_inputs(self.cfg)]

    def _score_many(self, loans):
        X = self._prepared_many(loans)
        pdh = self.sc.predict_pd(X)
        s = self.sc.score(X)
        return pdh, s, self.sc.band(s), X

    def _band_rank(self, band):
        return self.sc.band_names.index(band)              # riskiest = 0, safest = last

    def score_loan(self, loan):
        loan = validate_fields(loan)
        self._check_required(loan)
        pdh, s, bands, X = self._score_many([loan])
        p, s, band = float(pdh[0]), float(s[0]), str(bands[0])
        reasons = []
        for item in self.sc.reason_codes(X, top_n=3)[0]:
            m = re.match(r"(\w+) \(-([\d.]+) pts\)", item)
            if not m:
                continue
            name, lost = m.group(1), float(m.group(2))
            value = loan.get(name, DEFAULTS.get(name, X[name].iloc[0] if name in X else ""))
            if name in FLAG_LABELS:
                value = "yes" if int(X[name].iloc[0]) == 1 else "no"
            elif value == "X" or (name in UNKNOWN_IS_ZERO and float(value) == 0):
                value = "unknown (not recorded)"
            r = {"factor": LABELS.get(name, name), "field": name, "value": value, "points_lost": round(lost, 1)}
            med = (self.baseline or {}).get("features", {}).get(name, {}).get("median")
            if med is not None:
                r["portfolio_median"] = round(med, 1)
            reasons.append(r)
        facts = {"score": int(round(s)), "pd_percent": round(p * 100, 1), "risk_band": band,
                 "band_position": f"{self._band_rank(band) + 1} of {len(self.sc.band_names)}, where 1 is the riskiest",
                 "reasons": reasons, "inputs": loan}
        unk = [LABELS[k] for k in sorted(UNKNOWN_IS_ZERO) if float(loan.get(k, 1)) == 0]
        if unk:
            facts["unknown_inputs"] = unk
        b = self.baseline or {}
        if b.get("performance"):
            facts["portfolio_average_pd_percent"] = round(b["performance"]["mean_pd"] * 100, 1)
        rate = (b.get("band_bad_rates") or {}).get(band)
        if rate is not None and not np.isnan(rate):
            facts["band_historical_bad_rate_percent"] = round(rate * 100, 1)
        return facts

    # ------------------------------------------------------------------ what-if
    def what_if(self, loan, specs, label=""):
        loan = validate_fields(loan)
        self._check_required(loan)
        specs = sc_mod.validate_changes(specs)
        if not specs:
            raise NeedsInfo("Tell me what to change, for example 'what if the loan amount was 10% higher?'.")
        new, applied, notes = sc_mod.apply_changes(loan, specs)
        before, after = self.score_loan(loan), self.score_loan(new)
        band_change = "unchanged" if before["risk_band"] == after["risk_band"] else \
            f"{before['risk_band']} to {after['risk_band']}"
        if abs(after["score"] - before["score"]) <= 2 and applied:
            notes = notes + ["Scorecard inputs are grouped into bins, so a change that stays inside the same bin barely moves the score."]
        return {"scenario": label or "; ".join(sc_mod.describe_change(s) for s in specs),
                "applied": applied, "notes": notes,
                "before_score": before["score"], "after_score": after["score"],
                "score_change": after["score"] - before["score"],
                "before_pd_percent": before["pd_percent"], "after_pd_percent": after["pd_percent"],
                "pd_change_points": round(after["pd_percent"] - before["pd_percent"], 1),
                "before_band": before["risk_band"], "after_band": after["risk_band"], "band_change": band_change}

    # ------------------------------------------------------------------ goal seek
    def what_it_takes(self, loan, target=None):
        """For each lever on its own, the smallest change that reaches the target (band, score or PD)."""
        loan = validate_fields(loan)
        self._check_required(loan)
        cur = self.score_loan(loan)
        target = target or {}
        names = self.sc.band_names
        if "band" in target:
            letter = target["band"].upper()
            match = [n for n in names if n.upper().startswith(letter + " ")]
            if not match:
                raise GovernanceError(f"There is no risk band '{letter}'.")
            goal = {"description": f"risk band {match[0]}", "band": match[0]}
        elif "score" in target:
            goal = {"description": f"a score of {int(target['score'])}", "score": float(target["score"])}
        elif "pd_percent" in target:
            goal = {"description": f"a PD of {target['pd_percent']:g}% or lower", "pd": target["pd_percent"] / 100}
        else:
            rank = self._band_rank(cur["risk_band"])
            if rank >= len(names) - 1:
                return {"already_best": True, "current_score": cur["score"], "current_band": cur["risk_band"]}
            goal = {"description": f"the next better risk band ({names[rank + 1]})", "band": names[rank + 1]}

        def reached(pdh, s, bands):
            if "band" in goal:
                return np.array([self._band_rank(b) >= self._band_rank(goal["band"]) for b in bands])
            if "score" in goal:
                return s >= goal["score"]
            return pdh <= goal["pd"]

        if reached(np.array([cur["pd_percent"] / 100]), np.array([cur["score"]]), [cur["risk_band"]])[0]:
            return {"goal": goal["description"], "already_met": True, "current_score": cur["score"],
                    "current_band": cur["risk_band"], "current_pd_percent": cur["pd_percent"]}

        cs0, ltv0, dti0 = float(loan["CreditScore"]), float(loan["LTV"]), float(loan["DTI"])
        grids = {"CreditScore": [float(x) for x in np.arange(cs0 + 1, 851) if x <= 850][::1] if cs0 > 0 else [float(x) for x in range(300, 851, 5)],
                 "LTV": [round(float(x), 1) for x in np.arange(ltv0 - 0.5, 19.5, -0.5)],
                 "DTI": [round(float(x), 1) for x in np.arange(dti0 - 0.5, 4.5, -0.5)] if dti0 > 0 else [float(x) for x in range(65, 4, -1)]}
        levers = []
        for field, grid in grids.items():
            unknown = field in UNKNOWN_IS_ZERO and float(loan[field]) == 0
            if not grid:
                levers.append({"lever": LABELS[field], "field": field, "reachable": False})
                continue
            cands = [{**loan, field: v} for v in grid]
            pdh, s, bands, _ = self._score_many(cands)
            ok = np.where(reached(pdh, s, bands))[0]
            if len(ok) == 0:
                levers.append({"lever": LABELS[field], "field": field, "reachable": False})
                continue
            v = grid[ok[0]]
            item = {"lever": LABELS[field], "field": field, "reachable": True, "from": loan[field], "to": v,
                    "change": round(v - float(loan[field]), 1), "score_after": int(round(s[ok[0]]))}
            if unknown:                                   # nothing to "move" from: report the value that would be needed
                item.update(not_recorded=True, **{"from": "not recorded"}, change=None)
            if field == "LTV":
                new_upb = float(loan["OrigUPB"]) * v / ltv0
                item["equivalent_loan_amount"] = round(new_upb, 0)
                item["extra_down_payment"] = round(float(loan["OrigUPB"]) - new_upb, 0)
            levers.append(item)
        out = {"goal": goal["description"], "current_score": cur["score"], "current_band": cur["risk_band"],
               "current_pd_percent": cur["pd_percent"], "levers": levers,
               "note": "Each lever is moved on its own with everything else unchanged."}
        if "band" in goal:
            edge = self.sc.band_cutoffs_[self._band_rank(goal["band"]) - 1]      # score must exceed this
            out["points_needed"] = int(np.floor(edge - cur["score"])) + 1
        elif "score" in goal:
            out["points_needed"] = int(np.ceil(goal["score"] - cur["score"]))
        return out

    # ------------------------------------------------------------------ compare
    def compare_loans(self, loan_a, loan_b):
        loan_a, loan_b = validate_fields(loan_a, "loan A"), validate_fields(loan_b, "loan B")
        self._check_required(loan_a, "loan A")
        self._check_required(loan_b, "loan B")
        pdh, s, bands, X = self._score_many([loan_a, loan_b])
        n = len(self.sc.selected_)
        losses = []
        for i in range(2):
            d = {}
            for item in self.sc.reason_codes(X.iloc[[i]], top_n=n)[0]:
                m = re.match(r"(\w+) \(-([\d.]+) pts\)", item)
                if m:
                    d[m.group(1)] = float(m.group(2))
            losses.append(d)
        drivers = []
        for f in self.sc.selected_:
            la, lb = losses[0].get(f, 0.0), losses[1].get(f, 0.0)
            va = loan_a.get(f, DEFAULTS.get(f, ""))
            vb = loan_b.get(f, DEFAULTS.get(f, ""))
            drivers.append({"factor": LABELS.get(f, f), "field": f, "loan_a_value": va, "loan_b_value": vb,
                            "loan_a_points_lost": round(la, 1), "loan_b_points_lost": round(lb, 1),
                            "advantage_points": round(la - lb, 1)})        # positive = loan B better
        drivers.sort(key=lambda d: -abs(d["advantage_points"]))
        return {"loan_a": {"score": int(round(s[0])), "pd_percent": round(float(pdh[0]) * 100, 1), "risk_band": str(bands[0])},
                "loan_b": {"score": int(round(s[1])), "pd_percent": round(float(pdh[1]) * 100, 1), "risk_band": str(bands[1])},
                "score_gap_b_minus_a": int(round(s[1])) - int(round(s[0])),
                "pd_gap_points": round((float(pdh[1]) - float(pdh[0])) * 100, 1),
                "drivers": [d for d in drivers if abs(d["advantage_points"]) >= 0.5][:3]}

    # --------------------------------------------------------------- monitoring
    def _monitoring(self):
        f = self.monitoring_dir / "batch_monitoring.csv"
        if not f.exists():
            return None, None
        feat = self.monitoring_dir / "feature_psi.csv"
        return pd.read_csv(f), (pd.read_csv(feat) if feat.exists() else None)

    def drift_status(self):
        summary, feat = self._monitoring()
        if summary is None:
            return {"available": False}
        th = thresholds(self.cfg)
        last = summary.iloc[-1]
        top = []
        if feat is not None:
            t = feat[(feat.batch == last["batch"]) & feat.in_model].sort_values("psi", ascending=False).head(3)
            top = [{"feature": LABELS.get(r.feature, r.feature), "psi": round(float(r.psi), 3), "status": r.status}
                   for r in t.itertuples()]
        facts = {"available": True, "batches": int(len(summary)), "latest_batch": str(last["batch"]),
                 "status": str(last["status"]), "score_psi": round(float(last["score_psi"]), 3),
                 "first_batch_score_psi_watch": first_breach(summary, "score_psi", th["psi_watch"]),
                 "first_batch_score_psi_alert": first_breach(summary, "score_psi", th["psi_alert"]),
                 "top_shifted_features": top,
                 "share_riskiest_band_percent": round(float(last["share_riskiest_band"]) * 100, 1)}
        if not np.isnan(last["auc"]):
            facts.update({"latest_auc": round(float(last["auc"]), 3),
                          "calibration_ratio": round(float(last["calibration_ratio"]), 2)})
        if self.baseline and self.baseline.get("performance"):
            facts["baseline_auc"] = round(self.baseline["performance"]["auc"], 3)
        return facts

    def model_performance(self):
        out = {"available": bool(self.baseline and self.baseline.get("performance"))}
        if out["available"]:
            p = self.baseline["performance"]
            out.update({"baseline_auc": round(p["auc"], 3), "baseline_gini": round(p["gini"], 3),
                        "baseline_ks": round(p["ks"], 3), "baseline_bad_rate_percent": round(p["bad_rate"] * 100, 1)})
        d = self.drift_status()
        if d.get("available") and "latest_auc" in d:
            out.update({"latest_batch": d["latest_batch"], "latest_auc": d["latest_auc"],
                        "calibration_ratio": d["calibration_ratio"]})
        return out

    def model_health(self, focus="all"):
        if focus == "performance":
            return {"focus": "performance", **self.model_performance()}
        d = self.drift_status()
        if focus == "drift":
            return {"focus": "drift", **d}
        p = self.model_performance()
        return {"focus": "all", **d, **{k: v for k, v in p.items() if k not in d}}

    # ----------------------------------------------------------- documentation
    def model_info(self, topic):
        """Fixed, reviewed explanations of the model. Numbers come from config and the baseline."""
        c, b = self.cfg, self.baseline or {}
        perf = b.get("performance", {})
        sc_cfg, th = c["scorecard"], thresholds(c)
        bands = c["risk_bands"]
        shares, prev = [], 0
        for x in bands:
            shares.append(x["upper_pct"] - prev)
            prev = x["upper_pct"]
        sel = ", ".join(LABELS.get(f, f) for f in self.sc.selected_)
        text = {
            "data": "The model was built on real Freddie Mac mortgage loans from a single origination quarter (Q1 1999)"
                    + (f", using a development sample of {b['n_reference']:,} loans" if b.get("n_reference") else "")
                    + ". It is a portfolio demonstration, not a production lending model.",
            "target": f"It predicts serious delinquency: a loan being {c['target']['serious_months']} or more months late"
                      + (f" (about {perf['bad_rate'] * 100:.1f}% of loans)" if perf else "") + ".",
            "method": f"A weight-of-evidence scorecard with logistic regression. Score = offset + factor x ln((1 - PD) / PD): "
                      f"{sc_cfg['base_score']} points at {sc_cfg['base_odds']}:1 good-to-bad odds, and every "
                      f"{sc_cfg['pdo']} points doubles the odds. A higher score means lower risk.",
            "features": f"The model uses these inputs: {sel}. All are known when the loan is approved.",
            "excluded": "Outcome and leakage columns, post-origination fields, and location or seller fields are excluded "
                        "as model inputs, including postal code, state, MSA and seller. Location is kept only as a monitoring segment.",
            "bands": "Loans are split into five risk bands by score percentile of the development data: "
                     + ", ".join(f"{x['name']} ({s}%)" for x, s in zip(bands, shares))
                     + ". Band E is the riskiest. Each band's observed bad rate is shown with a loan's explanation.",
            "psi": f"PSI (Population Stability Index) measures how much the mix of incoming loans has moved away from the "
                   f"development data. Below {th['psi_watch']} is stable, {th['psi_watch']} to {th['psi_alert']} is a watch, "
                   f"above {th['psi_alert']} is a significant shift.",
            "auc": "AUC measures how well the model ranks risky loans above safe ones: 0.5 is a coin flip and 1.0 is perfect."
                   + (f" On held-out data it is {perf['auc']:.3f}." if perf else ""),
            "ks": "KS is the largest gap between the cumulative share of bad loans and good loans when loans are sorted by score. "
                  "Bigger means better separation."
                  + (f" On held-out data it is {perf['ks']:.3f}." if perf else ""),
            "calibration": "Calibration asks whether predicted PDs match reality. The monitoring ratio is actual bad rate divided by "
                           f"average predicted PD: 1.0 is perfect, and a gap beyond {int(th['calib_watch'] * 100)}% is flagged.",
            "limitations": "One origination vintage, so no out-of-time validation. A random train/test split is optimistic. "
                           "No income, employment or bureau history in the data. Drift results use simulated batches. "
                           "Fairness metrics by segment are not yet computed.",
            "fairness": "Protected characteristics and location fields are not model inputs. Segment-level fairness metrics are not yet computed, "
                        "so no claim of fairness testing is made.",
            "pd": "PD is the model's estimated probability that the loan becomes seriously delinquent. It is a probability for a "
                  "group of similar loans, not a prediction about one borrower.",
        }
        topic = topic if topic in text else "method"
        return {"topic": topic, "answer": text[topic]}


TOPIC_KEYWORDS = [
    ("limitations", r"limitation|limits|weakness|caveat|not good at|shortcoming"),
    ("fairness", r"fair|bias|discriminat\w* (?:against|by)|protected"),
    ("excluded", r"exclud|not use|leakage|forbidden"),
    ("psi", r"\bpsi\b|population stability"),
    ("auc", r"\bauc\b|\broc\b|\bgini\b"),
    ("ks", r"\bks\b|kolmogorov"),
    ("calibration", r"calibrat"),
    ("bands", r"\bbands?\b"),
    ("target", r"target|what does .*predict|what is being predicted|delinquen"),
    ("features", r"feature|variable|input|which (?:fields|columns)"),
    ("data", r"\bdata\b|dataset|trained on|freddie|vintage"),
    ("pd", r"\bpd\b|probability of default"),
    ("method", r"method|how (?:is|was|does) .*(?:built|trained|calculated|work)|scorecard|woe|logistic|points"),
]


def topic_from_text(text):
    t = text.lower()
    return next((k for k, pat in TOPIC_KEYWORDS if re.search(pat, t)), "method")
