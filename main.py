"""
main.py — CLI entry point for the AI Agent Coordination & Decision Engine.

Usage:
    python main.py
    python main.py --query "Analyse our Q2 sales performance and suggest actions"
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError with rich and special characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from memory.shared_memory import SharedMemory
from workflows.agent_workflow import AgentWorkflow

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║      Enterprise Customer Support & Resolution Engine             ║
║      Powered by Multi-Agent Coordination                         ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    console.print(BANNER, style="bold cyan")


def print_plan(plan: dict):
    console.print(Rule("[bold yellow]Execution Plan[/bold yellow]"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Task", min_width=40)
    table.add_column("Agent", width=12)
    table.add_column("Depends On", width=12)

    for task in plan.get("tasks", []):
        deps = ", ".join(str(d) for d in task.get("depends_on", [])) or "—"
        table.add_row(
            str(task.get("id", "")),
            task.get("description", ""),
            task.get("agent", ""),
            deps,
        )
    console.print(table)
    console.print()


def print_response(response: str):
    console.print(Rule("[bold green]Final Response[/bold green]"))
    console.print(Markdown(response))
    console.print()


def run_query(query: str, memory: SharedMemory):
    workflow = AgentWorkflow(memory=memory)

    console.print(
        Panel(f"[bold white]{query}[/bold white]", title="[cyan]User Request[/cyan]", border_style="cyan")
    )

    with console.status("[bold green]Agents working …[/bold green]", spinner="dots"):
        result = workflow.run(query)

    print_plan(result["plan"])
    print_response(result["response"])

    console.print(
        f"[dim]Session ID: {result['session_id']} | "
        f"Tasks completed: {len(result['task_results'])}[/dim]"
    )
    return result


def interactive_mode():
    memory = SharedMemory()
    console.print("[dim]Type your query, or 'exit' to quit. Type 'history' to view message history.[/dim]\n")

    while True:
        query = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if query.lower() == "history":
            history = memory.get_history()
            if not history:
                console.print("[dim]No history yet.[/dim]")
            for msg in history:
                role = msg["role"].upper()
                agent = f" ({msg['agent']})" if msg.get("agent") else ""
                console.print(f"[bold]{role}{agent}:[/bold] {msg['content'][:200]}")
            continue

        try:
            run_query(query, memory)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent Coordination & Decision Engine — CLI"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query and exit.",
    )
    args = parser.parse_args()

    print_banner()

    if args.query:
        memory = SharedMemory()
        try:
            run_query(args.query, memory)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            sys.exit(1)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
