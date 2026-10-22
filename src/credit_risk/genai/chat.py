"""Command-line entry point for the assistant.

    python -m credit_risk.genai.chat                                                # interactive, loans looked up by ID
    python -m credit_risk.genai.chat --loan-id F199Q1290567 --ask "why is this loan risky?"
    python -m credit_risk.genai.chat --loan examples/loan_example.json              # interactive, loan from a file
    python -m credit_risk.genai.chat --loan examples/loan_example.json --ask "why is this loan risky?"
    python -m credit_risk.genai.chat --loan examples/loan_example.json --loan-b examples/loan_example_b.json
    python -m credit_risk.genai.chat --report                                      # monitoring report
    python -m credit_risk.genai.chat --loan examples/loan_example.json --loan-note # one-loan note

In the chat you can also type  :loan A file.json   :loan B file.json   :show   (blank line to quit).
Add --no-llm to force the rule-based fallback; set ANTHROPIC_API_KEY to use Claude.
"""
import argparse
import json

from . import governance as gov
from .agent import Assistant
from .llm import check, get_client
from .loans import LoanStore
from .report import build_loan_report, build_report
from .tools import ModelService


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loan", help="JSON file with loan A")
    ap.add_argument("--loan-b", help="JSON file with loan B (for comparisons)")
    ap.add_argument("--loan-id", help="look up loan A in the loan file by its loan ID")
    ap.add_argument("--loan-id-b", help="look up loan B in the loan file by its loan ID")
    ap.add_argument("--raw", help="loan file to look IDs up in (default: paths.raw in config)")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("--report", action="store_true", help="write reports/monitoring_report.md")
    ap.add_argument("--loan-note", action="store_true", help="write a one-loan note to reports/")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="if a Claude call fails, use the rule-based reader instead of stopping")
    ap.add_argument("--check-llm", action="store_true", help="test the Claude connection and exit")
    ap.add_argument("--monitoring", default="monitoring")
    a = ap.parse_args()

    if a.check_llm:
        ok, msg = check()
        print(("PASS: " if ok else "FAIL: ") + msg)
        return
    service = ModelService.from_artifacts(monitoring_dir=a.monitoring)
    llm = None if a.no_llm else get_client()
    store = LoanStore(a.raw)
    bot = Assistant(service, llm, store=store, fallback=a.fallback)
    print(f"[mode] routing + explanations: {'Claude (' + llm.model + ')' if llm else 'rule-based fallback (no API key)'}")
    if a.loan:
        bot.session.loans["A"] = gov.validate_fields(json.load(open(a.loan)))
    if a.loan_b:
        bot.session.loans["B"] = gov.validate_fields(json.load(open(a.loan_b)))
    for flag, slot in ((a.loan_id, "A"), (a.loan_id_b, "B")):
        if flag:
            bot.session.loans[slot], info = store.get(flag)
            print(f"Loaded loan {info['loan_id']} as loan {slot}.")

    if a.report:
        build_report(service, bot, "reports/monitoring_report.md")
        print("Wrote reports/monitoring_report.md")
        return
    if a.loan_note:
        path, _ = build_loan_report(service, bot, bot.session.loans["A"])
        print(f"Wrote {path}")
        return

    def show(r):
        print(f"[{r['intent']} via {r['intent_source']}, answer via {r['answer_source']}]\n{r['answer']}")
        if r.get("llm_error"):
            print(f"[Claude call failed, used the fallback instead -> {r['llm_error']}]")

    if a.ask:
        show(bot.ask(a.ask))
        return
    print("Ask a question (blank line to quit). Name a loan by its ID (for example 'why is loan F199Q1290567 risky?'),\n"
          "say 'a random loan', or type details by hand. Commands: :loan A file.json, :show")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        if q.startswith(":loan"):
            _, which, path = q.split(maxsplit=2)
            bot.session.loans[which.upper()] = gov.validate_fields(json.load(open(path)))
            print(f"Loaded loan {which.upper()}.")
            continue
        if q == ":show":
            print(json.dumps(bot.session.loans, indent=2))
            continue
        r = bot.ask(q)
        show(r)
        if r["end"]:
            break


if __name__ == "__main__":
    main()
