import json
from collections import defaultdict
from datetime import datetime


def generate_report(evaluated_results: list, output_path: str = "audit_report") -> dict:
    by_category = defaultdict(lambda: {"pass": 0, "fail": 0, "failures": []})

    for r in evaluated_results:
        cat = r["category"]
        if r["passed"]:
            by_category[cat]["pass"] += 1
        else:
            by_category[cat]["fail"] += 1
            by_category[cat]["failures"].append(r)

    total = len(evaluated_results)
    total_pass = sum(1 for r in evaluated_results if r["passed"])
    total_fail = total - total_pass
    pass_rate = round(total_pass / total * 100, 1) if total else 0

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total": total,
            "passed": total_pass,
            "failed": total_fail,
            "pass_rate_pct": pass_rate,
        },
        "by_category": {
            cat: {
                "pass": v["pass"],
                "fail": v["fail"],
                "fail_rate_pct": round(v["fail"] / (v["pass"] + v["fail"]) * 100, 1),
            }
            for cat, v in by_category.items()
        },
        "failures": [r for r in evaluated_results if not r["passed"]],
    }

    with open(f"{output_path}.json", "w") as f:
        json.dump(report, f, indent=2)

    _write_markdown(report, by_category, f"{output_path}.md")
    return report


def _write_markdown(report: dict, by_category: dict, path: str):
    s = report["summary"]
    lines = [
        "# Agent Security Audit Report",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total prompts | {s['total']} |",
        f"| Passed | {s['passed']} |",
        f"| Failed | {s['failed']} |",
        f"| Pass rate | {s['pass_rate_pct']}% |",
        "",
        "## Results by Category",
        "",
        "| Category | Pass | Fail | Fail Rate |",
        "|----------|------|------|-----------|",
    ]

    for cat, v in sorted(by_category.items(), key=lambda x: -x[1]["fail"]):
        total_cat = v["pass"] + v["fail"]
        fail_rate = round(v["fail"] / total_cat * 100, 1) if total_cat else 0
        lines.append(f"| {cat} | {v['pass']} | {v['fail']} | {fail_rate}% |")

    failures = [r for r in report["failures"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [
                f"### [{r['severity'].upper()}] {r['category']} — `{r['id']}`",
                "",
                f"**Prompt:** {r['prompt']}",
                "",
                f"**Response:** {r.get('response', '(none)')}",
                "",
                f"**Verdict:** {r['reason']}",
                "",
                f"**Violated policy:** {r['violated_policy']}",
                "",
                "---",
                "",
            ]
    else:
        lines += ["", "## Failures", "", "_No failures detected._", ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))
