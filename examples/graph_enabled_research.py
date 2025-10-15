"""Run wiki/blog/news agents with the LightRAG graph builder enabled."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import httpx
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


console = Console()
BASE_URL = os.environ.get("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
REPORTS_DIR = Path("reports")
GRAPH_ROOT = Path("graph_outputs")

GRAPH_LLM_MODEL = os.environ.get("GRAPH_LLM_MODEL")
GRAPH_LLM_BASE_URL = os.environ.get("GRAPH_LLM_BASE_URL", "http://localhost:8000/v1")
GRAPH_LLM_API_KEY = os.environ.get("GRAPH_LLM_API_KEY")
GRAPH_EMBED_MODEL = os.environ.get("GRAPH_EMBED_MODEL")
GRAPH_EMBED_BASE_URL = os.environ.get("GRAPH_EMBED_BASE_URL", "http://localhost:8001/v1")
GRAPH_EMBED_API_KEY = os.environ.get("GRAPH_EMBED_API_KEY")
GRAPH_COLLECTOR_URL = os.environ.get("GRAPH_COLLECTOR_URL")


AGENTS = {
    "wiki": {
        "name": "Wikipedia",
        "config": {"search_sites": "en.wikipedia.org"},
        "emoji": "📖",
    },
    "blog": {
        "name": "Blog",
        "config": {"search_inurl": "blog"},
        "emoji": "✍️",
    },
    "news": {
        "name": "News",
        "config": {"search_inurl": "news"},
        "emoji": "📰",
    },
}


def create_assistant(agent_id: str) -> str:
    payload = {
        "name": f"researcher_{agent_id}",
        "graph_id": "slm_deep_researcher",
        "if_exists": "do_nothing",
    }
    response = httpx.post(f"{BASE_URL}/assistants", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["assistant_id"]


def build_graph_config(workspace: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "graph_builder_enabled": True,
        "graph_workspace": str(workspace),
    }
    if GRAPH_LLM_MODEL:
        config["graph_llm_model"] = GRAPH_LLM_MODEL
    if GRAPH_LLM_BASE_URL:
        config["graph_llm_base_url"] = GRAPH_LLM_BASE_URL
    if GRAPH_LLM_API_KEY:
        config["graph_llm_api_key"] = GRAPH_LLM_API_KEY
    if GRAPH_EMBED_MODEL:
        config["graph_embedding_model"] = GRAPH_EMBED_MODEL
    if GRAPH_EMBED_BASE_URL:
        config["graph_embedding_base_url"] = GRAPH_EMBED_BASE_URL
    if GRAPH_EMBED_API_KEY:
        config["graph_embedding_api_key"] = GRAPH_EMBED_API_KEY
    if GRAPH_COLLECTOR_URL:
        config["graph_collector_url"] = GRAPH_COLLECTOR_URL
    return config


async def run_research(
    agent_id: str,
    agent_config: Dict[str, Any],
    topic: str,
    timestamp: str,
    workspace: Path,
    progress: Progress,
    task_id: int,
) -> Dict[str, Any]:
    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Starting...")
    assistant_id = create_assistant(agent_id)
    workspace.mkdir(parents=True, exist_ok=True)

    config_payload = {
        **agent_config["config"],
        **build_graph_config(workspace),
    }

    payload = {
        "assistant_id": assistant_id,
        "input": {"research_topic": topic},
        "on_completion": "delete",
        "stream_mode": ["events"],
        "config": {"configurable": config_payload},
    }

    summary = None
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{BASE_URL}/runs/stream", json=payload) as response:
            response.raise_for_status()

            buffer = ""
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                fragment = line[5:].strip()
                if not fragment or fragment == "[DONE]":
                    continue

                buffer += fragment
                try:
                    event = json.loads(buffer)
                    buffer = ""
                except json.JSONDecodeError:
                    continue
                kind = event.get("event")
                name = event.get("name")

                if kind != "on_chain_end":
                    continue

                if name == "generate_query":
                    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Searching...")
                elif name == "web_research":
                    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Gathering sources...")
                elif name == "graph_builder":
                    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Building graph...")
                elif name == "summarize_sources":
                    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Summarizing...")
                elif name == "finalize_summary":
                    summary = event.get("data", {}).get("output", {}).get("running_summary")
                    progress.update(task_id, description=f"{agent_config['emoji']} {agent_config['name']}: Complete!")

    return {
        "agent_id": agent_id,
        "agent_name": agent_config["name"],
        "emoji": agent_config["emoji"],
        "summary": summary or "No summary generated",
        "config": config_payload,
        "workspace": workspace,
    }


def save_report(result: Dict[str, Any], topic: str, timestamp: str) -> Path:
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


def describe_graph(workspace: Path) -> str:
    graph_file = workspace / "graph_chunk_entity_relation.graphml"
    if not graph_file.exists():
        return "Graph data not generated"

    graph = nx.read_graphml(graph_file)
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    sample = list(graph.nodes(data=True))[:5]
    lines = [f"Nodes: {nodes}, Edges: {edges}"]
    for node_id, data in sample:
        lines.append(f"- {node_id}: {data.get('entity_type')} :: {data.get('description', '')[:80]}")
    return "\n".join(lines)


async def main() -> None:
    topic = "Do you have any advice for handling the 4-month sleep regression?"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    REPORTS_DIR.mkdir(exist_ok=True)
    GRAPH_ROOT.mkdir(exist_ok=True)
    shared_workspace = GRAPH_ROOT / timestamp

    console.print(
        Panel.fit(
            f"[bold white]{topic}[/bold white]",
            title="[bold cyan]Graph-Enabled Research[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        tasks = []
        for agent_id, agent_config in AGENTS.items():
            task_id = progress.add_task(
                f"{agent_config['emoji']} {agent_config['name']}: Initializing...",
                total=None,
            )
            tasks.append(run_research(agent_id, agent_config, topic, timestamp, shared_workspace, progress, task_id))

        results = await asyncio.gather(*tasks)

    console.print()
    console.print("[bold green]✓[/bold green] All agents completed!")
    console.print()

    table = Table(title="Research Reports Generated", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="cyan")
    table.add_column("Summary Path", style="green")
    table.add_column("Graph Overview", overflow="fold")

    shared_graph_summary = describe_graph(shared_workspace)
    first_graph = True

    for result in results:
        report_path = save_report(result, topic, timestamp)
        graph_summary = shared_graph_summary if first_graph else "Shared graph (see above)"
        table.add_row(result["emoji"] + " " + result["agent_name"], str(report_path), graph_summary)
        first_graph = False

    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
