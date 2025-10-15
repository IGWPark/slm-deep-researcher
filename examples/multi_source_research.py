"""Multi-source research example using parallel agents."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


console = Console()
BASE_URL = os.environ.get("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
REPORTS_DIR = Path("reports")


AGENTS = {
    "academic": {
        "name": "Academic Research",
        "config": {"search_sites": ".edu"},
        "emoji": "📚",
    },
    "blog": {
        "name": "Blog",
        "config": {"search_inurl": "blog"},
        "emoji": "✍️",
    },
    "community": {
        "name": "Community Discussion",
        "config": {"search_sites": "reddit.com/r/Parenting"},
        "emoji": "💬",
    },
    "wiki": {
        "name": "Wikipedia",
        "config": {"search_sites": "en.wikipedia.org"},
        "emoji": "📖",
    },
    "news": {
        "name": "News Articles",
        "config": {"search_inurl": "news"},
        "emoji": "📰",
    },
}


def create_assistant(assistant_id: str) -> str:
    """Creates or retrieves assistant."""
    payload = {
        "name": f"researcher_{assistant_id}",
        "graph_id": "slm_deep_researcher",
        "if_exists": "do_nothing",
    }
    response = httpx.post(
        f"{BASE_URL}/assistants",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["assistant_id"]


async def run_research(
    agent_id: str,
    agent_config: Dict[str, Any],
    topic: str,
    progress: Progress,
    task_id: int,
) -> Dict[str, Any]:
    """Runs research for a single agent."""
    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Starting...")

    assistant_id = create_assistant(agent_id)

    payload = {
        "assistant_id": assistant_id,
        "input": {"research_topic": topic},
        "on_completion": "delete",
        "stream_mode": ["events"],
        "config": {"configurable": agent_config["config"]},
    }

    summary = None

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/runs/stream",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                event = json.loads(line[5:])
                kind = event.get("event")
                name = event.get("name")

                if kind == "on_chain_end":
                    if name == "generate_query":
                        query = event.get("data", {}).get("output", {}).get("search_query")
                        if query:
                            progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Searching...")

                    elif name == "web_research":
                        progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Analyzing sources...")

                    elif name == "summarize_sources":
                        progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Summarizing...")

                    elif name == "reflect_on_summary":
                        progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Refining...")

                    elif name == "finalize_summary":
                        summary = event.get("data", {}).get("output", {}).get("running_summary")
                        progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Complete!")

    return {
        "agent_id": agent_id,
        "agent_name": agent_config["name"],
        "emoji": agent_config["emoji"],
        "summary": summary or "No summary generated",
        "config": agent_config["config"],
    }


def save_report(result: Dict[str, Any], topic: str, timestamp: str) -> Path:
    """Saves research report as markdown."""
    REPORTS_DIR.mkdir(exist_ok=True)

    agent_slug = result["agent_id"].replace("_", "-")
    filename = f"{timestamp}_{agent_slug}.md"
    filepath = REPORTS_DIR / filename

    config_str = ", ".join(f"{k}={v}" for k, v in result["config"].items())

    content = f"""# {result['emoji']} {result['agent_name']}

**Topic:** {topic}
**Generated:** {timestamp}
**Configuration:** {config_str}

---

{result['summary']}
"""

    filepath.write_text(content)
    return filepath


async def main():
    topic = "Do you have any advice for handling the 4-month sleep regression?"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    console.print(Panel.fit(
        f"[bold white]{topic}[/bold white]",
        title="[bold cyan]Multi-Source Research[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        tasks = []
        task_ids = {}

        for agent_id, agent_config in AGENTS.items():
            task_id = progress.add_task(
                f"{agent_config['emoji']} {agent_config['name']}: Initializing...",
                total=None
            )
            task_ids[agent_id] = task_id

            tasks.append(
                run_research(agent_id, agent_config, topic, progress, task_id)
            )

        results = await asyncio.gather(*tasks)

    console.print()
    console.print("[bold green]✓[/bold green] All agents completed!")
    console.print()

    table = Table(title="Research Reports Generated", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="cyan")
    table.add_column("File", style="green")

    for result in results:
        filepath = save_report(result, topic, timestamp)
        table.add_row(
            f"{result['emoji']} {result['agent_name']}",
            str(filepath)
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Reports saved to: {REPORTS_DIR.absolute()}[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise
