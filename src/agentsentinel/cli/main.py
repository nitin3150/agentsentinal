import click
from importlib.metadata import version, PackageNotFoundError

try:
    _version = version("agentsentinel-ai")
except PackageNotFoundError:
    _version = "dev"


@click.group()
@click.version_option(_version, prog_name="agentsentinel")
def cli() -> None:
    """Agent Sentinel — inspect, optimize, and stress-test AI agents."""


from agentsentinel.cli.commands.inspect import inspect_cmd       # noqa: E402
from agentsentinel.cli.commands.optimize import optimize_cmd     # noqa: E402
from agentsentinel.cli.commands.stress_test import stress_test_cmd  # noqa: E402
from agentsentinel.cli.commands.audit import audit_cmd           # noqa: E402

cli.add_command(inspect_cmd)
cli.add_command(optimize_cmd)
cli.add_command(stress_test_cmd)
cli.add_command(audit_cmd)
