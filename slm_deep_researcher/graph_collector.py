"""Minimal aggregator API for LightRAG graph building."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Configuration
from .kg import build_graph_workspace

app = FastAPI()
_DATA_ROOT = Path("graph_collector_store")
_DATA_ROOT.mkdir(exist_ok=True)


class IngestRequest(BaseModel):
    workspace: str = Field(..., description="Logical workspace identifier")
    topic: str = Field(..., description="Research topic or identifier")
    results: List[Dict[str, Any]] = Field(..., description="List of search result payloads")


class BuildRequest(BaseModel):
    workspace: str = Field(..., description="Workspace to build")
    topic: Optional[str] = Field(default=None, description="Topic override")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration overrides")
    clear: bool = Field(default=False, description="Delete buffered data after build")


def _workspace_file(workspace: str) -> Path:
    safe = workspace.replace("/", "_")
    return _DATA_ROOT / f"{safe}.json"


def _load_workspace(workspace: str) -> Dict[str, Any]:
    file_path = _workspace_file(workspace)
    if not file_path.exists():
        return {"results": [], "processed": []}
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_workspace(workspace: str, data: Dict[str, Any]) -> None:
    file_path = _workspace_file(workspace)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


@app.post("/ingest", status_code=202)
async def ingest(request: IngestRequest) -> Dict[str, Any]:
    data = _load_workspace(request.workspace)
    existing_urls = {item.get("url") for item in data.get("results", []) if item.get("url")}

    appended = 0
    for result in request.results:
        url = result.get("url")
        if url and url in existing_urls:
            continue
        data.setdefault("results", []).append(result)
        if url:
            existing_urls.add(url)
        appended += 1

    data["topic"] = request.topic
    _save_workspace(request.workspace, data)
    return {"status": "accepted", "appended": appended}


def _build_configuration(overrides: Dict[str, Any], workspace: str) -> Configuration:
    overrides = dict(overrides or {})
    if not overrides.get("graph_workspace"):
        default_root = Path("graph_outputs_server")
        default_root.mkdir(exist_ok=True)
        overrides["graph_workspace"] = str(default_root / workspace)

    base = Configuration.from_env().model_dump()
    base.update(overrides)
    base["graph_builder_enabled"] = True
    return Configuration(**base)


@app.post("/build")
async def build(request: BuildRequest) -> Dict[str, Any]:
    data = _load_workspace(request.workspace)
    results: List[Dict[str, Any]] = data.get("results", [])
    if not results:
        return {"processed": []}

    overrides = dict(request.config or {})
    cfg = _build_configuration(overrides, workspace=request.workspace)
    effective_topic = request.topic or data.get("topic") or request.workspace
    processed_urls = await build_graph_workspace(
        documents=results,
        topic=effective_topic,
        cfg=cfg,
        processed_urls=data.get("processed", []),
    )

    data.setdefault("processed", [])
    if processed_urls:
        data["processed"] = sorted({*data["processed"], *processed_urls})
    if request.clear:
        data["results"] = []
    _save_workspace(request.workspace, data)
    return {"processed": processed_urls}


@app.post("/clear")
async def clear(workspace: str) -> Dict[str, Any]:
    file_path = _workspace_file(workspace)
    if file_path.exists():
        file_path.unlink()
    return {"status": "cleared"}


def run(host: str = "127.0.0.1", port: int = 8085) -> None:
    import uvicorn

    uvicorn.run("slm_deep_researcher.graph_collector:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
