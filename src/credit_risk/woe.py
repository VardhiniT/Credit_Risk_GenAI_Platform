"""Weight-of-Evidence (WOE) binning, written from scratch so every step is explainable.

Convention used everywhere in this project:
    WOE = ln( share of GOOD loans in the bin / share of BAD loans in the bin )
    -> positive WOE = safer than average, negative WOE = riskier than average.
    IV  = sum( (good_share - bad_share) * WOE )

Rules:
  * numeric: start from quantile bins, merge small bins, enforce a monotone trend (optional),
    then cap at max_bins. Missing values get their own bin.
  * categorical: rare categories are grouped, similar categories are merged to max_bins.
  * unseen categories at scoring time fall into the UNKNOWN bin if there is one, else WOE = 0.
"""
import numpy as np
import pandas as pd

SMOOTH = 0.5
MISSING = "MISSING"


def _woe_iv(good, bad, G, B):
    gs = (good + SMOOTH) / (G + SMOOTH)
    bs = (bad + SMOOTH) / (B + SMOOTH)
    woe = np.log(gs / bs)
    return woe, (gs - bs) * woe


def _iv_of(bins):
    G = sum(b["good"] for b in bins)
    B = sum(b["bad"] for b in bins)
    g = np.array([b["good"] for b in bins], float)
    bd = np.array([b["bad"] for b in bins], float)
    return float(_woe_iv(g, bd, G, B)[1].sum())


def _rate(b):
    n = b["good"] + b["bad"]
    return b["bad"] / n if n else 0.0


def _merge(bins, i):
    """Merge bin i with bin i+1 (adjacent)."""
    a, b = bins[i], bins[i + 1]
    m = {"good": a["good"] + b["good"], "bad": a["bad"] + b["bad"]}
    if "upper" in a:
        m["upper"] = b["upper"]
    if "members" in a:
        m["members"] = a["members"] + b["members"]
    return bins[:i] + [m] + bins[i + 2:]


def _merge_small(bins, min_n):
    while len(bins) > 1:
        sizes = [b["good"] + b["bad"] for b in bins]
        i = int(np.argmin(sizes))
        if sizes[i] >= min_n:
            break
        if i == 0:
            j = 0
        elif i == len(bins) - 1:
            j = i - 1
        else:   # merge with the neighbour whose bad rate is closest
            j = i - 1 if abs(_rate(bins[i]) - _rate(bins[i - 1])) <= abs(_rate(bins[i]) - _rate(bins[i + 1])) else i
        bins = _merge(bins, j)
    return bins


def _enforce_monotone(bins, increasing):
    while True:
        rates = [_rate(b) for b in bins]
        bad_i = next((i for i in range(len(rates) - 1)
                      if (rates[i + 1] < rates[i] if increasing else rates[i + 1] > rates[i])), None)
        if bad_i is None or len(bins) <= 1:
            return bins
        bins = _merge(bins, bad_i)


def _cap_bins(bins, max_bins):
    while len(bins) > max_bins:
        diffs = [abs(_rate(bins[i]) - _rate(bins[i + 1])) for i in range(len(bins) - 1)]
        bins = _merge(bins, int(np.argmin(diffs)))
    return bins


class WOEBinner:
    def __init__(self, max_bins=8, min_bin_pct=0.05, monotonic=True, n_initial=20):
        self.max_bins, self.min_bin_pct = max_bins, min_bin_pct
        self.monotonic, self.n_initial = monotonic, n_initial
        self.specs_, self.tables_, self.iv_ = {}, {}, {}

    # ------------------------------------------------------------------ fit
    def fit(self, df, y, numeric, categorical):
        y = np.asarray(y).astype(int)
        G, B = int((y == 0).sum()), int((y == 1).sum())
        min_n = int(self.min_bin_pct * len(y))
        for c in numeric:
            self._fit_numeric(c, df[c], y, G, B, min_n)
        for c in categorical:
            self._fit_categorical(c, df[c], y, G, B, min_n)
        return self

    def _fit_numeric(self, c, s, y, G, B, min_n):
        x = pd.to_numeric(s, errors="coerce").to_numpy(float)
        miss = np.isnan(x)
        xv, yv = x[~miss], y[~miss]
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, self.n_initial + 1)[1:-1]))
        uppers = np.append(qs, np.inf)
        idx = np.searchsorted(uppers, xv, side="left")
        bins = [{"upper": float(u),
                 "good": int(((idx == k) & (yv == 0)).sum()),
                 "bad": int(((idx == k) & (yv == 1)).sum())} for k, u in enumerate(uppers)]
        bins = [b for b in bins if b["good"] + b["bad"] > 0]
        bins = _merge_small(bins, min_n)
        if self.monotonic and len(bins) > 2:
            up, down = _enforce_monotone(bins, True), _enforce_monotone(bins, False)
            bins = up if _iv_of(up) >= _iv_of(down) else down
        bins = _cap_bins(bins, self.max_bins)
        spec = {"kind": "numeric", "uppers": [b["upper"] for b in bins]}
        if miss.any():
            spec["has_missing"] = True
            bins = bins + [{"upper": None, "good": int(((y == 0) & miss).sum()), "bad": int(((y == 1) & miss).sum())}]
        else:
            spec["has_missing"] = False
        self._finish(c, spec, bins, G, B)

    def _fit_categorical(self, c, s, y, G, B, min_n):
        x = s.astype("string").fillna("UNKNOWN").astype(object).to_numpy()
        cats = pd.Series(x).value_counts()
        rare = [k for k, n in cats.items() if n < min_n and k != "UNKNOWN"]
        x = np.where(np.isin(x, rare), "OTHER", x)
        tab = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["count", "sum"])
        tab["rate"] = tab["sum"] / tab["count"]
        bins = [{"members": [k], "good": int(r["count"] - r["sum"]), "bad": int(r["sum"])}
                for k, r in tab.sort_values("rate").iterrows()]
        bins = _cap_bins(bins, self.max_bins)
        spec = {"kind": "categorical", "groups": [b["members"] for b in bins], "rare": rare}
        self._finish(c, spec, bins, G, B)

    def _finish(self, c, spec, bins, G, B):
        g = np.array([b["good"] for b in bins], float)
        bd = np.array([b["bad"] for b in bins], float)
        woe, ivc = _woe_iv(g, bd, G, B)
        spec["woe"] = [float(w) for w in woe]
        self.specs_[c] = spec
        self.iv_[c] = float(ivc.sum())
        n = g + bd
        self.tables_[c] = pd.DataFrame({
            "feature": c, "bin": range(len(bins)), "label": self._labels(spec),
            "count": n.astype(int), "pct": n / n.sum(), "good": g.astype(int), "bad": bd.astype(int),
            "bad_rate": bd / np.where(n == 0, 1, n), "woe": woe, "iv_contrib": ivc})

    @staticmethod
    def _labels(spec):
        if spec["kind"] == "categorical":
            rare = spec["rare"]
            shown = ",".join(map(str, rare[:3])) + ("..." if len(rare) > 3 else "")

            def name(m):
                return f"OTHER[{shown}]" if m == "OTHER" and rare else str(m)
            return [", ".join(name(m) for m in g) for g in spec["groups"]]
        labs, lo = [], "-inf"
        for u in spec["uppers"]:
            labs.append(f"({lo}, {'inf' if np.isinf(u) else format(u, 'g')}]")
            lo = format(u, "g")
        if spec["has_missing"]:
            labs.append(MISSING)
        return labs

    # ------------------------------------------------------------ transform
    def bin_index(self, df, c):
        spec = self.specs_[c]
        if spec["kind"] == "numeric":
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            miss = np.isnan(x)
            idx = np.searchsorted(np.array(spec["uppers"]), np.where(miss, 0, x), side="left")
            idx = np.minimum(idx, len(spec["uppers"]) - 1)
            if spec["has_missing"]:
                idx = np.where(miss, len(spec["uppers"]), idx)
            else:   # unseen missing at scoring time -> neutral: use median-ish middle bin
                idx = np.where(miss, len(spec["uppers"]) // 2, idx)
            return idx
        lookup = {m: k for k, g in enumerate(spec["groups"]) for m in g}
        unseen = lookup.get("UNKNOWN", None)
        x = df[c].astype("string").fillna("UNKNOWN").astype(object).to_numpy()
        x = np.where(np.isin(x, spec["rare"]), "OTHER", x)
        return np.array([lookup.get(v, unseen if unseen is not None else -1) for v in x])

    def transform(self, df, features=None):
        out = {}
        for c in (features or list(self.specs_)):
            woe = np.array(self.specs_[c]["woe"] + [0.0])    # last slot (-1) = neutral
            out[c] = woe[self.bin_index(df, c)]
        return pd.DataFrame(out, index=df.index)

    def iv_table(self):
        return (pd.Series(self.iv_, name="iv").sort_values(ascending=False)
                .rename_axis("feature").reset_index())

    def bins_table(self):
        return pd.concat(self.tables_.values(), ignore_index=True)
