"""
Terminal Demo for Milestones 1-3: Agent Coordination & Decision Engine
This script demonstrates the core functionality of the agent framework without requiring
the web dashboard or Docker (Milestones 4-5).
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from backend.orchestration.workflow_engine import engine
from backend.orchestration.message_bus import message_bus
from backend.core.config import AgentRole

console = Console()

# We will capture and print events in real-time to the terminal
def handle_agent_event(event):
    if "started" in event.event_type:
        color = "cyan"
    elif "completed" in event.event_type:
        color = "green"
    elif "error" in event.event_type:
        color = "red"
    elif "tool" in event.event_type:
        color = "yellow"
    else:
        color = "white"
        
    role = event.agent_role.upper()
    console.print(f"[{color}][{role}][/] {event.message}")


async def run_demo():
    console.print(Panel.fit(
        "AI Agent Coordination & Decision Engine\n[green]Milestones 1-3 Demonstration[/]",
        border_style="blue"
    ))
    
    # Subscribe to message bus to print live agent chatter
    message_bus.subscribe_all(handle_agent_event)
    
    console.print("\n[bold magenta]Welcome to the AI Agent Coordination Engine (Interactive Mode)[/bold magenta]")
    console.print("[dim]Type 'exit' or 'quit' to stop.[/dim]\n")

    while True:
        try:
            business_request = input("\nEnter your business request: ").strip()
            if not business_request:
                continue
            if business_request.lower() in ["exit", "quit"]:
                break
                
            # Create the workflow state (we use 'custom' so the planner decides the flow)
            console.print("[dim]Initializing multi-agent workflow...[/dim]")
            state = engine.create_workflow(business_request, "custom")
            
            # Execute the workflow
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Agents are collaborating...", total=None)
                
                # Run execution
                result = await engine.execute_workflow(state.id)
                
            console.print("\n[bold green]=== WORKFLOW COMPLETED ===[/bold green]")
            
            # Print the final output from the Executor Agent
            if "execution_output" in result["final_context"]:
                console.print(Panel(
                    result["final_context"]["execution_output"], 
                    title="Final Output & Actions Taken", 
                    border_style="green"
                ))
            else:
                console.print("[yellow]Workflow processed, but no formatted final execution output was returned.[/yellow]")
                
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[red]Error during execution: {e}[/red]")
            
    console.print("\n[bold blue]=== SESSION ENDED ===[/bold blue]")

if __name__ == "__main__":
    # Ensure our required database tables exist
    from backend.tools.enterprise_tools import _initialize_db
    _initialize_db()
    
    # Run the async demo
    asyncio.run(run_demo())
