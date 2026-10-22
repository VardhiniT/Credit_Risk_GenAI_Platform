"""Guardrails for the GenAI layer.

Principles (each one is enforced in code and covered by a test):

  1. The LLM never calculates. Every number comes from the scorecard or the monitoring tables.
  2. Inputs are checked: only approved loan fields, in valid ranges. Names, addresses, postal codes,
     and anything else not a model input are rejected, so personal data never reaches the LLM.
  3. Outputs are checked: every number in an explanation must exist in the facts it was given,
     and decision language ("approve", "decline") is blocked. A failed check falls back to a fixed template.
  4. Everything is logged (question, route, facts, answer, which checks passed) in an append-only audit file.
"""
import hashlib
import json
import re
import time
from pathlib import Path

PROMPT_VERSION = "v1"


class GovernanceError(ValueError):
    """Raised when a request breaks an input rule. The message is safe to show to the user."""


class NeedsInfo(GovernanceError):
    """The request is fine but a detail is missing. The message asks the user for it."""


NUMERIC_RANGES = {            # field: (min, max)
    "CreditScore": (300, 850), "DTI": (0, 65), "LTV": (1, 105), "OrigUPB": (10_000, 2_000_000),
    "OrigInterestRate": (1.0, 15.0), "MIP": (0, 40), "Units": (1, 4),
}
CATEGORY_VALUES = {
    "FirstTimeHomebuyer": {"Y", "N", "X"}, "Occupancy": {"O", "I", "S", "X"}, "Channel": {"R", "B", "C", "T", "X"},
    "PPM": {"Y", "N", "X"}, "PropertyType": {"SF", "CO", "PU", "MH", "CP", "LH", "X"},
    "LoanPurpose": {"P", "C", "N", "X"}, "NumBorrowers": {"1", "2", "X"},
}
# Context fields are NOT model inputs. They only help translate a scenario (for example an income change)
# into the model inputs it affects (DTI). They are never passed to the model.
CONTEXT_RANGES = {"annual_income": (10_000, 5_000_000)}
UNKNOWN_IS_ZERO = {"CreditScore", "DTI"}                  # same convention as data.clean(): 0 = unknown, plus an explicit flag
MODEL_FIELDS = set(NUMERIC_RANGES) | set(CATEGORY_VALUES)
ALLOWED_FIELDS = MODEL_FIELDS | set(CONTEXT_RANGES)
LABELS = {
    "CreditScore": "credit score", "DTI": "debt-to-income ratio (DTI)", "LTV": "loan-to-value ratio (LTV)",
    "OrigUPB": "original loan amount", "OrigInterestRate": "interest rate", "MIP": "mortgage insurance coverage",
    "Units": "number of units", "FirstTimeHomebuyer": "first-time homebuyer status", "Occupancy": "occupancy type",
    "Channel": "origination channel", "PPM": "prepayment penalty", "PropertyType": "property type",
    "LoanPurpose": "loan purpose", "NumBorrowers": "number of borrowers",
    "annual_income": "annual income", "property_value": "property value",
    "cs_unknown": "missing credit score", "dti_unknown": "missing DTI", "ltv_gt80": "LTV above 80%",
}


def validate_fields(fields, label="loan"):
    """Return a clean dict with typed values. Reject unknown fields and out-of-range values."""
    if not isinstance(fields, dict) or not fields:
        raise GovernanceError(f"No {label} details were provided.")
    bad = sorted(set(fields) - ALLOWED_FIELDS)
    if bad:
        raise GovernanceError(
            f"Field(s) not accepted: {', '.join(bad)}. Only these model inputs are accepted: "
            f"{', '.join(sorted(ALLOWED_FIELDS))}. Personal or location details are never used.")
    clean = {}
    for k, v in fields.items():
        if k in NUMERIC_RANGES or k in CONTEXT_RANGES:
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise GovernanceError(f"{k} must be a number.")
            lo, hi = NUMERIC_RANGES.get(k) or CONTEXT_RANGES[k]
            if k in UNKNOWN_IS_ZERO and x == 0:                 # 0 means "not recorded" in this dataset; the model has a flag for it
                clean[k] = 0.0
                continue
            if not lo <= x <= hi:
                raise GovernanceError(
                    f"{LABELS.get(k, k)} of {x:,.4g} is outside the range the model was built on ({lo:,} to {hi:,}), "
                    f"so I can't score that reliably.")
            clean[k] = x
        else:
            x = str(v).strip().upper()
            if x not in CATEGORY_VALUES[k]:
                raise GovernanceError(f"{k} must be one of {sorted(CATEGORY_VALUES[k])} (got '{v}').")
            clean[k] = x
    return clean


# ----------------------------------------------------------------------------- request screening
# Cheap regex guards that run BEFORE any LLM sees the text (defence in depth).
_INJECTION = re.compile(
    r"ignore (?:all |any |your |the )?(?:previous|prior|above|earlier)|disregard (?:your|the|all|any)|system prompt|"
    r"reveal (?:your|the) (?:prompt|instructions|rules)|you are now|jailbreak|developer mode|pretend (?:to be|you are)|"
    r"forget (?:your|all|the) (?:rules|instructions)", re.I)
_SENSITIVE = re.compile(
    r"\b(race|racial|ethnic\w*|religio\w*|gender|sex|female|male|women|woman|national origin|nationality|immigra\w*|"
    r"marital|married|pregnan\w*|disabilit\w*|zip ?code|postal|neighbou?rhood|redlin\w*|ssn|social security|"
    r"date of birth|home address|borrower'?s? name|older borrowers?|younger borrowers?|age of the (?:borrower|applicant))\b", re.I)
_ASKS_DECISION = re.compile(
    r"\b(?:should|can|could|would|do|shall) (?:we|i|you|they)\s+(?:\w+\s+){0,2}?(?:approve|decline|reject|deny|lend|grant|fund|underwrite)\b|"
    r"\b(?:approve|decline|reject|deny|turn down|fund)\s+(?:this|the|that|her|his|their|it)\b|"
    r"\bgo ahead (?:and|with) (?:lend|approv)|\bpass or fail\b|\b(?:is|are) (?:this|the|she|he) (?:borrower )?(?:eligible|qualified|creditworthy)\b|"
    r"\bwill (?:this|the) (?:loan|application) (?:be )?(?:approved|declined|rejected)", re.I)
_LEGAL = re.compile(r"\b(legal|legally|lawsuit|sue|compliance|compliant|regulat\w+|ecoa|cfpb|illegal|lawful|liable|liability|attorney)\b", re.I)


def screen_request(text):
    """Return None if the text is fine, otherwise one of: injection, sensitive, advice."""
    if _INJECTION.search(text):
        return "injection"
    if _SENSITIVE.search(text):
        return "sensitive"
    if _ASKS_DECISION.search(text) or _LEGAL.search(text):
        return "advice"
    return None


# ----------------------------------------------------------------------------- output checks
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_ID_LIKE = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{8,14}\b")
_DECISION = re.compile(r"\b(approve[ds]?|approval|decline[ds]?|deny|denied|reject(?:ed)?|should (?:lend|not lend))\b", re.I)


def _flatten_numbers(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass


def ungrounded_numbers(text, facts):
    """Numbers in `text` that do not appear (rounded) in `facts`. Small whole numbers 0-5 are allowed (counts, 'top 3')."""
    allowed = []
    _flatten_numbers(facts, allowed)
    bad = []
    text = _ID_LIKE.sub(" ", text)                       # loan IDs such as F199Q1178517 are identifiers, not figures
    for m in _NUM.findall(text):
        try:
            x = float(m.replace(",", "").rstrip("."))
        except ValueError:
            continue
        if abs(x) <= 5 and float(x).is_integer():
            continue
        ok = any(abs(abs(x) - abs(a)) <= max(0.051, 0.006 * abs(a)) or
                 abs(abs(x) - abs(a) * 100) <= max(0.051, 0.006 * abs(a) * 100) or
                 abs(abs(x) - abs(a) / 100) <= max(0.0006, 0.006 * abs(a) / 100) for a in allowed)
        if not ok:
            bad.append(m)
    return bad


def check_answer(text, facts):
    """Return a dict of check results. `passed` is True only if every check passes."""
    nums = ungrounded_numbers(text, facts)
    decision = bool(_DECISION.search(text))
    return {"numbers_grounded": not nums, "ungrounded": nums, "no_decision_language": not decision,
            "passed": (not nums) and (not decision)}


# ----------------------------------------------------------------------------- audit log
class AuditLog:
    def __init__(self, path="logs/genai_audit.jsonl"):
        self.path = Path(path)

    def write(self, **record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "prompt_version": PROMPT_VERSION, **record}
        if "question" in record:
            record["question_sha256"] = hashlib.sha256(record["question"].encode()).hexdigest()[:16]
            record["question"] = record["question"][:500]
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
