from pathlib import Path

import click

from agentsentinel.cli.loader import load_agent
from agentsentinel.cli.output import (
    console,
    print_compliance_panel,
    print_risk_table,
    print_scores_panel,
    print_stress_failures_table,
    print_stress_summary_panel,
)
from agentsentinel.cli.utils import check_env, parse_compliance
from agentsentinel.sentinel import AgentSentinel


@click.command("audit")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--compliance", default="", callback=parse_compliance, is_eager=True,
              help="Comma-separated standards: hipaa,soc2,owasp,pii,All")
@click.option("--policies", default="", type=click.Path(), help="Path to policy PDF")
@click.option("--threshold", default=80.0, type=float, show_default=True,
              help="Pass rate % required to consider audit complete")
@click.option("--max-iterations", default=3, type=int, show_default=True,
              help="Max optimize → re-test cycles")
@click.option("--output-dir", default=None, type=click.Path(),
              help="Directory to save audit result files")
def audit_cmd(source, compliance, policies, threshold, max_iterations, output_dir):
    """Full audit loop: inspect → stress-test → optimize → repeat until threshold met."""
    check_env()
    agent = load_agent(source)
    sentinel = AgentSentinel()

    with console.status("Running full audit...", spinner="dots"):
        result = sentinel.audit(
            agent,
            compliance=compliance,
            policies=policies,
            pass_threshold=threshold,
            max_iterations=max_iterations,
        )

    profile = result["profile"]
    report = result["report"]
    iteration = result["iteration"]

    print_risk_table(profile.risk_flags)
    print_scores_panel(profile)
    print_compliance_panel(profile.compliance_results)
    print_stress_failures_table(report)
    print_stress_summary_panel(report)

    pass_rate = report["summary"]["pass_rate_pct"]
    color = "green" if pass_rate >= threshold else "red"
    console.print(
        f"\n[bold]Audit complete[/bold] — "
        f"iterations: {iteration}, "
        f"final pass rate: [{color}]{pass_rate}%[/{color}] "
        f"(threshold: {threshold}%)"
    )

    if output_dir:
        import json
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "audit_report.json").write_text(json.dumps(report, indent=2, default=str))
        console.print(f"[dim]Results saved → {output_dir}/audit_report.json[/dim]")
