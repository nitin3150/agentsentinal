from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def severity_color(severity: str) -> str:
    return {"high": "red", "medium": "yellow", "low": "green"}.get(severity.lower(), "white")


def print_profile_panel(profile) -> None:
    content = (
        f"[bold]Framework:[/bold] {profile.framework}\n"
        f"[bold]Domain:[/bold] {profile.domain or '(unset)'}\n"
        f"[bold]Prompt length:[/bold] {len(profile.system_prompt)} chars\n"
        f"[bold]Tools:[/bold] {len(profile.tool_definitions)}"
    )
    console.print(Panel(content, title="Agent Profile", border_style="blue"))


def print_risk_table(risk_flags: list) -> None:
    table = Table(title="Risk Flags", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Severity")
    table.add_column("Location")
    table.add_column("Suggestion")
    for flag in risk_flags:
        color = severity_color(flag.severity.value)
        table.add_row(
            flag.category.value,
            f"[{color}]{flag.severity.value}[/{color}]",
            flag.location,
            flag.suggestion,
        )
    console.print(table)


def print_scores_panel(profile) -> None:
    risk_color = severity_color(profile.overall_risk.value)
    inj_color = severity_color(profile.injection_surface.value)
    content = (
        f"[bold]Overall Risk:[/bold] [{risk_color}]{profile.overall_risk.value.upper()}[/{risk_color}]\n"
        f"[bold]Persona Clarity:[/bold] {profile.persona_clarity_score}/10\n"
        f"[bold]Scope Definition:[/bold] {profile.scope_definition_score}/10\n"
        f"[bold]Tone Consistency:[/bold] {profile.tone_consistency_score}/10\n"
        f"[bold]Injection Surface:[/bold] [{inj_color}]{profile.injection_surface.value}[/{inj_color}]\n"
        f"[bold]Baseline Score:[/bold] {profile.estimated_baseline_score}/100"
    )
    console.print(Panel(content, title="Scores", border_style="blue"))


def print_compliance_panel(compliance_results: dict) -> None:
    if not compliance_results:
        return
    lines = []
    for standard, result in compliance_results.items():
        status = "[green]PASS[/green]" if result.compliant else "[red]FAIL[/red]"
        lines.append(f"[bold]{standard.upper()}:[/bold] {status}")
        if not result.compliant:
            for v in result.violations:
                color = severity_color(v.severity.value)
                lines.append(f"  • [{color}]{v.description}[/{color}]")
    console.print(Panel("\n".join(lines), title="Compliance", border_style="blue"))


def print_change_log_panel(change_log: list) -> None:
    lines = [f"[bold]{e.field}:[/bold] {e.reason}" for e in change_log]
    content = "\n".join(lines) if lines else "[dim]No changes applied.[/dim]"
    console.print(Panel(content, title="Changes Applied", border_style="green"))


def print_stress_failures_table(report: dict) -> None:
    failures = report.get("failures", [])
    if not failures:
        console.print("[green]No failures.[/green]")
        return
    table = Table(title="Failures", show_header=True, header_style="bold red")
    table.add_column("Category", style="cyan")
    table.add_column("Severity")
    table.add_column("Prompt", max_width=50)
    table.add_column("Verdict", max_width=50)
    for f in failures:
        color = severity_color(f.get("severity", "medium"))
        table.add_row(
            f.get("category", ""),
            f"[{color}]{f.get('severity', '')}[/{color}]",
            f.get("prompt", "")[:100],
            f.get("reason", ""),
        )
    console.print(table)


def print_stress_summary_panel(report: dict) -> None:
    s = report["summary"]
    rate = s["pass_rate_pct"]
    color = "green" if rate >= 80 else "yellow" if rate >= 60 else "red"
    content = (
        f"[bold]Pass Rate:[/bold] [{color}]{rate}%[/{color}]\n"
        f"[bold]Passed:[/bold] {s['passed']} / {s['total']}\n"
        f"[bold]Failed:[/bold] {s['failed']}\n"
        f"[bold]Skipped:[/bold] {s['skipped']}"
    )
    console.print(Panel(content, title="Stress Test Summary", border_style="blue"))


def print_audit_iteration_header(iteration: int, max_iterations: int) -> None:
    console.rule(f"[bold blue]Audit Iteration {iteration}/{max_iterations}[/bold blue]")
