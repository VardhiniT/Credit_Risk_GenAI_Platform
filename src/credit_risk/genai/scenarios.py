"""Scenario language: turn a sentence like "loan up 10% and income down 50k" into structured changes,
then apply them to a loan, translating non-model quantities into the model inputs they affect.

A change is {"field": ..., "op": "set" | "add" | "pct", "value": number}.
    set  -> new value            ("credit score was 740")
    add  -> add a signed amount  ("DTI up by 5", "income down 50k")
    pct  -> signed percentage    ("loan up 10%")

Translation rules (stated to the user as assumptions, never hidden):
  * property value is not a model input. It is inferred as loan amount / LTV, and LTV follows any change to the
    loan amount or the property value.
  * income is not a model input. A change in income changes DTI in inverse proportion, assuming monthly debt
    payments stay the same. It needs `annual_income` in the loan details.
"""
import re

from .governance import LABELS, MODEL_FIELDS, NUMERIC_RANGES, GovernanceError, NeedsInfo, validate_fields

CHANGE_FIELDS = sorted(MODEL_FIELDS | {"annual_income", "property_value"})
OPS = ("set", "add", "pct")

# keyword -> field, tried in order (most specific first)
_KEYWORDS = [
    ("property_value", r"(?:property|home|house|collateral)\s*(?:is |are |was )?(?:value|price|worth|valued)s?\b|(?:house|home) prices?|\bworth\b"),
    ("annual_income", r"\bincome\b|\bsalary\b|\bearnings\b"),
    ("LTV", r"\bltv\b|loan[- ]to[- ]value"),
    ("DTI", r"\bdti\b|debt[- ]to[- ]income"),
    ("CreditScore", r"credit\s*score|\bfico\b|credit rating"),
    ("OrigInterestRate", r"interest\s*rate|mortgage rate|\brates?\b"),
    ("MIP", r"mortgage insurance|\bmip\b"),
    ("OrigUPB", r"loan\s*amount|loan size|principal|\bbalance\b|\bupb\b|borrowing|\bloan\b(?![- ]to)"),
]
_POS = r"increas\w*|\bup\b|higher|\bmore\b|rais\w*|\bris(?:e|es|ing|en)\b|\brose\b|grow\w*|grew|bigger|larger|jump\w*|\bhike\w*|\bgain\w*|improv\w*"
_NEG = r"decreas\w*|\bdown\b|lower\w*|\bless\b|reduc\w*|\bdrop\w*|\bfall\w*|\bfell\b|\bcut\b|smaller|declin\w*|\bdip\w*|worsen\w*|deteriorat\w*"
_NUM = re.compile(r"(\d[\d,]*\.?\d*)\s*(%|percent|k\b|thousand|m\b|million|bps|basis points|pts|points)?", re.I)
_SPLIT = re.compile(r"\s*(?:,(?!\d)|;|\band\b|\bbut\b|\bwhile\b|\bplus\b|&)\s*", re.I)

# multiplicative wording with no number: pattern -> signed percentage
_MULT = [(r"\bdoubl\w*|\btwice\b", 100.0), (r"\btripl\w*", 200.0), (r"\b(?:in|by) half\b|\bhalv\w*|\bhalf\b", -50.0),
         (r"\bthree[- ]quarters?\b", 75.0), (r"\ba quarter\b|\bone quarter\b", 25.0), (r"\ba third\b|\bone third\b", 100 / 3),
         (r"\ba fifth\b|\bone fifth\b", 20.0), (r"\ba tenth\b|\bone tenth\b", 10.0), (r"\ban eighth\b", 12.5)]

# illustrative stress scenarios (assumptions are shown to the user and can be edited here)
STRESS_PRESETS = {
    "downturn": ("Illustrative downturn: property value -10%, credit score -30, DTI +3",
                 [{"field": "property_value", "op": "pct", "value": -10},
                  {"field": "CreditScore", "op": "add", "value": -30},
                  {"field": "DTI", "op": "add", "value": 3}]),
    "severe downturn": ("Illustrative severe downturn: property value -20%, credit score -50, DTI +5",
                        [{"field": "property_value", "op": "pct", "value": -20},
                         {"field": "CreditScore", "op": "add", "value": -50},
                         {"field": "DTI", "op": "add", "value": 5}]),
}


def find_preset(text):
    t = text.lower()
    if re.search(r"severe|worst|extreme", t) and re.search(r"stress|downturn|recession", t):
        return "severe downturn"
    if re.search(r"stress|downturn|recession|housing crash|market crash", t):
        return "downturn"
    return None


def skipped_fields(text):
    """Fields the user mentioned changing (a field word plus a direction word) that the rules could not turn into a change."""
    done = {c["field"] for c in parse_changes_rules(text)}
    out = []
    for clause in _SPLIT.split(text.lower()):
        field = next((f for f, pat in _KEYWORDS if re.search(pat, clause)), None)
        if field and field not in done and (re.search(_POS, clause) or re.search(_NEG, clause)) and field not in out:
            out.append(field)
    return out


def parse_changes_rules(text):
    """Rule-based extraction of changes from a question. Returns a list of change dicts."""
    out = {}
    for clause in _SPLIT.split(text.lower()):
        field = next((f for f, pat in _KEYWORDS if re.search(pat, clause)), None)
        if field is None:
            continue
        pos, neg = re.search(_POS, clause), re.search(_NEG, clause)
        m = _NUM.search(clause)
        if not m:                                              # "doubles", "cut in half", "a third lower"
            mult = next((v for pat, v in _MULT if re.search(pat, clause)), None)
            if mult is None or (abs(mult) not in (100.0, 200.0, 50.0) and not (pos or neg)):
                continue
            sign = -1 if (neg and not pos) or mult < 0 else 1
            out[field] = {"field": field, "op": "pct", "value": sign * abs(mult)}
            continue
        raw, unit = float(m.group(1).replace(",", "").rstrip(".")), (m.group(2) or "").lower()
        scale = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(unit, 1.0)
        if unit in ("bps", "basis points"):
            scale = 0.01
        value = raw * scale
        sign = -1 if (neg and not pos) or (neg and pos and neg.start() < pos.start()) else 1
        sp = re.search(r"(?<![\w.])([+\-\u2212\u2013])\s*\d|\b(plus|minus)\s+\d", clause)   # "+15", "-30", "minus 30"
        explicit = None
        if sp:
            explicit = -1 if (sp.group(1) in ("-", "\u2212", "\u2013") if sp.group(1) else sp.group(2) == "minus") else 1
            sign = explicit
        has_to = re.search(r"\b(?:to|at|equal(?:s|ed)?|=)\b", clause) is not None
        has_by = re.search(r"\bby\b", clause) is not None
        directional = bool(pos or neg) or explicit is not None
        if unit in ("%", "percent") and directional and not has_to:
            op, v = "pct", sign * value
        elif directional and (has_by or not has_to):
            op, v = "add", sign * value
        else:
            op, v = "set", value
        out[field] = {"field": field, "op": op, "value": v}
    return list(out.values())


def validate_changes(changes):
    """Check a list of change dicts (from rules or from an LLM). Returns a clean list or raises GovernanceError."""
    clean = []
    if not isinstance(changes, list):
        raise GovernanceError("Changes must be a list.")
    for c in changes:
        if not isinstance(c, dict) or c.get("field") not in CHANGE_FIELDS or c.get("op") not in OPS:
            raise GovernanceError("Unsupported change requested.")
        if c["field"] in NUMERIC_RANGES or c["field"] in ("annual_income", "property_value"):
            try:
                v = float(c["value"])
            except (TypeError, ValueError):
                raise GovernanceError(f"{LABELS.get(c['field'], c['field'])} change must be a number.")
        else:                                              # categorical: only "set" makes sense
            if c["op"] != "set":
                raise GovernanceError(f"{LABELS.get(c['field'], c['field'])} can only be set to a value.")
            v = str(c["value"]).strip().upper()
        clean.append({"field": c["field"], "op": c["op"], "value": v})
    return clean


def _new_value(old, spec):
    if spec["op"] == "set":
        return spec["value"]
    if spec["op"] == "add":
        return old + spec["value"]
    return old * (1 + spec["value"] / 100.0)


def fmt_num(v, signed=False):
    """Readable number: 50,000 not 5e+04; 0.5 stays 0.5."""
    sign = "+" if signed and v > 0 else ""
    if abs(v) >= 1000:
        return f"{sign}{v:,.0f}"
    return f"{sign}{v:,.4g}"


def describe_change(spec):
    name = LABELS.get(spec["field"], spec["field"])
    v = spec["value"]
    if isinstance(v, str):
        return f"{name} = {v}"
    if spec["op"] == "set":
        return f"{name} = {fmt_num(v)}"
    if spec["op"] == "pct":
        return f"{name} {v:+g}%"
    return f"{name} {fmt_num(v, signed=True)}"


def apply_changes(loan, specs):
    """Return (new_loan, applied, notes). `applied` lists every model input that changed (including derived ones)."""
    specs = {s["field"]: s for s in specs}
    for f in ("CreditScore", "DTI"):
        if float(loan.get(f, 1)) == 0 and (f in specs and specs[f]["op"] != "set" or (f == "DTI" and "annual_income" in specs and "DTI" not in specs)):
            raise NeedsInfo(f"The {LABELS[f]} is not recorded for this loan, so I can't change it by an amount. Give me a value, "
                            f"for example '{'credit score is 680' if f == 'CreditScore' else 'DTI is 35'}'.")
    new, notes = dict(loan), []
    upb0, ltv0 = float(loan["OrigUPB"]), float(loan["LTV"])
    value0 = upb0 / (ltv0 / 100.0)

    upb1 = _new_value(upb0, specs["OrigUPB"]) if "OrigUPB" in specs else upb0
    value1 = _new_value(value0, specs["property_value"]) if "property_value" in specs else value0
    if value1 <= 0 or upb1 <= 0:
        raise GovernanceError("That change would make the loan amount or property value zero or negative.")
    derived = {}
    if "LTV" in specs:
        ltv1 = _new_value(ltv0, specs["LTV"])
    elif "OrigUPB" in specs or "property_value" in specs:
        ltv1 = upb1 / value1 * 100.0
        derived["LTV"] = ("follows from the loan amount and property value" if "property_value" in specs or "OrigUPB" in specs else "")
        notes.append(f"Property value is not a model input; it was inferred as loan amount / LTV (about ${value0:,.0f}) "
                     f"and LTV was recalculated from the new loan amount and property value.")
    else:
        ltv1 = ltv0
    new["OrigUPB"], new["LTV"] = round(upb1, 0), round(ltv1, 1)

    dti1 = float(loan["DTI"])
    if "annual_income" in specs:                         # income first: monthly debt stays the same, so DTI scales
        if "annual_income" not in loan:
            raise NeedsInfo("To translate an income change into the model's debt-to-income ratio I need the borrower's "
                            "current annual income, which is not in the loan file. Tell me, for example: 'income is 90000', then ask again.")
        inc0 = float(loan["annual_income"])
        inc1 = _new_value(inc0, specs["annual_income"])
        if inc1 <= 0:
            raise GovernanceError("That change would make income zero or negative.")
        dti1 = dti1 * inc0 / inc1
        derived["DTI"] = "follows from the income change"
        notes.append(f"Income is not a model input. DTI was scaled by old income / new income "
                     f"(${inc0:,.0f} to ${inc1:,.0f}), assuming monthly debt payments stay the same.")
        new["annual_income"] = round(inc1, 0)
    if "DTI" in specs:                                   # then any explicit DTI change is applied on top
        dti1 = _new_value(dti1, specs["DTI"])
        if "annual_income" in specs and specs["DTI"]["op"] != "set":
            derived["DTI"] = "income change first, then the DTI change on top"
        elif "annual_income" in specs:
            notes.append("You set DTI directly, so it overrides the DTI implied by the income change.")
    new["DTI"] = round(dti1, 1)

    for f in ("CreditScore", "OrigInterestRate", "MIP", "Units"):
        if f in specs:
            v = _new_value(float(loan.get(f, 0)), specs[f])
            new[f] = round(v) if f in ("CreditScore", "Units") else round(v, 3)
    for f, s in specs.items():
        if f not in NUMERIC_RANGES and f not in ("annual_income", "property_value"):   # categorical
            new[f] = s["value"]

    new = validate_fields(new)                          # range check on the RESULT, with a friendly message
    applied = []
    for f in ["CreditScore", "LTV", "DTI", "OrigUPB", "OrigInterestRate", "MIP", "Units"] + \
             [k for k in new if k not in NUMERIC_RANGES and k not in ("annual_income",)]:
        if f in new and (f not in loan or new[f] != (float(loan[f]) if f in NUMERIC_RANGES else loan[f])):
            applied.append({"input": LABELS.get(f, f), "field": f, "before": loan.get(f), "after": new[f],
                            "why": derived.get(f, "")})
    return new, applied, notes


# ----------------------------------------------------------------------------- goal parsing
_BAND = re.compile(r"\bband\s*([a-e])\b|\b([a-e])\s*-\s*(?:very )?(?:low|medium|high)\b", re.I)


def parse_target_rules(text):
    """What does the user want to reach? Returns {"band": "B"} / {"score": 600} / {"pd_percent": 5.0} / {}."""
    t = text.lower()
    m = _BAND.search(t)
    if m:
        return {"band": (m.group(1) or m.group(2)).upper()}
    m = re.search(r"(?:pd|probability|default rate|delinquency)\D{0,20}?(\d+(?:\.\d+)?)\s*%", t)
    if m:
        return {"pd_percent": float(m.group(1))}
    m = re.search(r"(?:score of|reach|get to|hit|at least|above|over)\s*(?:a )?(?:score )?(?:of )?(\d{3})\b(?!\s*(?:credit|fico))", t)
    if m and not re.search(r"credit score|fico", t[max(0, m.start() - 12):m.end() + 12]):
        return {"score": float(m.group(1))}
    return {}
