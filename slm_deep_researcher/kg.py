"""Utilities for building a LightRAG-compatible knowledge graph workspace."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from openai import AsyncOpenAI

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.operate import chunking_by_token_size
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    compute_mdhash_id,
    wrap_embedding_func_with_attrs,
)

from .config import Configuration
from .llm import resolve_model_name
from .token import ApproximateTokenizer
from .utils import fetch_raw_content

logger = logging.getLogger(__name__)

_PENDING_JOBS: Dict[str, List[asyncio.Task[List[str]]]] = {}
_PENDING_URLS: Dict[str, Set[str]] = {}

_TYPE_CANONICAL = {
    "person": "Person",
    "individual": "Person",
    "organization": "Organization",
    "organisation": "Organization",
    "company": "Organization",
    "institution": "Organization",
    "government": "Organization",
    "location": "Location",
    "city": "Location",
    "country": "Location",
    "region": "Location",
    "state": "Location",
    "concept": "Concept",
    "idea": "Concept",
    "topic": "Concept",
    "event": "Event",
    "conference": "Event",
    "artifact": "Artifact",
    "equipment": "Artifact",
    "device": "Artifact",
    "product": "Artifact",
    "tool": "Artifact",
}


def _split_entity_types(entity_types: str) -> List[str]:
    return [
        part.strip()
        for part in (entity_types or "").split(",")
        if part.strip()
    ]


def _normalize_name(name: str) -> str:
    return " ".join((name or "").strip().split())


def _normalize_type(raw: str) -> str:
    canonical = _TYPE_CANONICAL.get((raw or "").strip().lower())
    return canonical or "Other"


def _merge_source_field(existing: Optional[str], additional: Iterable[str]) -> str:
    tokens: Set[str] = {token for token in (existing or "").split(GRAPH_FIELD_SEP) if token}
    tokens.update(additional)
    return GRAPH_FIELD_SEP.join(sorted(tokens)) if tokens else ""


def schedule_graph_build(
    run_id: str,
    *,
    topic: str,
    documents: List[Dict[str, Any]],
    cfg: Configuration,
    processed_urls: Iterable[str],
) -> None:
    payload = copy.deepcopy(documents)
    pending_urls = {doc.get("url") for doc in payload if doc.get("url")}
    tracked = _PENDING_URLS.setdefault(run_id, set())
    tracked.update(pending_urls)
    processed_union = set(processed_urls or []) | tracked
    task = asyncio.create_task(
        build_graph_workspace(
            documents=payload,
            topic=topic,
            cfg=cfg,
            processed_urls=processed_union,
        )
    )
    _PENDING_JOBS.setdefault(run_id, []).append(task)


async def await_graph_build(run_id: str) -> List[str]:
    tasks = _PENDING_JOBS.pop(run_id, [])
    _PENDING_URLS.pop(run_id, None)
    processed: List[str] = []
    for task in tasks:
        try:
            urls = await task
            if urls:
                processed.extend(urls)
        except Exception as exc:
            logger.warning("Graph builder background task failed: %s", exc)
    return processed


@dataclass
class ChunkTask:
    source_id: str
    url: str
    title: str
    order: int
    tokens: int
    content: str


@dataclass
class ChunkResult:
    source_id: str
    url: str
    chunk_id: str
    order: int
    tokens: int
    content: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]


@dataclass
class GraphBuilder:
    config: Configuration
    tokenizer: ApproximateTokenizer = field(init=False)
    entity_types: List[str] = field(init=False)
    tuple_delimiter: str = field(init=False)
    completion_delimiter: str = field(init=False)
    examples_block: str = field(init=False)
    llm_client: AsyncOpenAI = field(init=False)
    embedding_client: AsyncOpenAI = field(init=False)
    embedding_func: Any = field(init=False)
    working_dir: str = field(init=False)

    def __post_init__(self) -> None:
        self.entity_types = _split_entity_types(self.config.graph_entity_types)
        if not self.entity_types:
            raise ValueError("graph_entity_types must include at least one type")

        self.tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        self.completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        context = {
            "tuple_delimiter": self.tuple_delimiter,
            "completion_delimiter": self.completion_delimiter,
            "entity_types": ", ".join(self.entity_types),
            "language": self.config.graph_language,
        }
        self.examples_block = "\n".join(
            example.format(**context) for example in PROMPTS["entity_extraction_examples"]
        )

        llm_base = self.config.graph_llm_base_url or self.config.llm_base_url
        llm_key = self.config.graph_llm_api_key or self.config.llm_api_key
        if not llm_base:
            raise ValueError("Graph builder requires graph_llm_base_url or llm_base_url.")
        self.llm_model = resolve_model_name(
            self.config.graph_llm_model or self.config.llm_model,
            base_url=llm_base,
            api_key=llm_key,
        )
        self.llm_client = AsyncOpenAI(base_url=llm_base, api_key=llm_key or "EMPTY")

        embed_base = self.config.graph_embedding_base_url
        embed_key = self.config.graph_embedding_api_key
        if not embed_base:
            raise ValueError("Graph builder requires graph_embedding_base_url.")
        self.embedding_model = resolve_model_name(
            self.config.graph_embedding_model,
            base_url=embed_base,
            api_key=embed_key,
        )
        self.embedding_client = AsyncOpenAI(base_url=embed_base, api_key=embed_key or "EMPTY")

        async def _embed(texts: List[str], **_: Any) -> np.ndarray:
            response = await self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            return np.array([item.embedding for item in response.data])

        _embed.embedding_dim = self.config.graph_embedding_dim
        self.embedding_func = wrap_embedding_func_with_attrs(
            embedding_dim=self.config.graph_embedding_dim
        )(_embed)

        self.tokenizer = ApproximateTokenizer()
        self.working_dir = os.path.abspath(self.config.graph_workspace)

    async def build(
        self,
        topic: str,
        raw_results: List[Dict[str, Any]],
        processed_urls: Set[str],
    ) -> List[str]:
        new_sources = [
            result for result in raw_results if result.get("url") and result["url"] not in processed_urls
        ]
        if not new_sources:
            return []

        rag = LightRAG(
            working_dir=self.working_dir,
            workspace="",
            llm_model_func=self._llm_stub,
            embedding_func=self.embedding_func,
            vector_db_storage_cls_kwargs={"cosine_better_than_threshold": self.config.graph_cosine_threshold},
        )
        await rag.initialize_storages()

        timestamp = int(time.time())
        topic_id = self._topic_node_id(topic)
        file_label = self.config.graph_file_label

        await self._upsert_topic_node(rag, topic_id, topic or "Untitled Topic", timestamp, file_label)

        chunk_entries: Dict[str, Dict[str, Any]] = {}
        entity_records: Dict[str, Dict[str, Any]] = {}
        relations_records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        processed: List[str] = []
        source_infos: Dict[str, Tuple[str, str]] = {}
        chunk_tasks: List["ChunkTask"] = []

        try:
            for result in new_sources:
                url = result["url"]
                title = result.get("title") or url
                raw_content = result.get("raw_content") or result.get("content") or fetch_raw_content(url)
                if not raw_content:
                    logger.debug("Skipping source without content: %s", url)
                    continue

                source_id = self._source_node_id(url)
                source_infos[url] = (source_id, title)

                for order, chunk in enumerate(self._split_chunks(raw_content)):
                    chunk_tasks.append(
                        ChunkTask(
                            source_id=source_id,
                            url=url,
                            title=title,
                            order=order,
                            tokens=chunk["tokens"],
                            content=chunk["content"],
                        )
                    )

            if not chunk_tasks:
                return []

            semaphore = asyncio.Semaphore(max(1, self.config.graph_max_concurrent_requests))
            chunk_results = await asyncio.gather(
                *(self._process_chunk(task, semaphore) for task in chunk_tasks)
            )

            urls_with_results: Set[str] = set()

            for result in chunk_results:
                if not result:
                    continue
                chunk_entries[result.chunk_id] = {
                    "content": result.content,
                    "tokens": result.tokens,
                    "chunk_order_index": result.order,
                    "source_id": result.source_id,
                    "full_doc_id": result.source_id,
                    "file_path": file_label,
                    "source_url": result.url,
                }
                self._merge_entities(result.entities, result.source_id, entity_records)
                self._merge_relations(result.relations, result.source_id, entity_records, relations_records)
                urls_with_results.add(result.url)

            for url in urls_with_results:
                source_id, title = source_infos[url]
                await self._upsert_source_node(
                    rag,
                    topic_id,
                    source_id,
                    title,
                    url,
                    timestamp,
                    file_label,
                )
                processed.append(url)

            await self._upsert_entities(rag, entity_records, file_label, timestamp)
            await self._upsert_relations(rag, relations_records, file_label, timestamp)

            if chunk_entries:
                await rag.text_chunks.upsert(chunk_entries)
                await rag.chunks_vdb.upsert(chunk_entries)

            entity_vdb_entries = self._build_entity_vectors(entity_records, file_label)
            if entity_vdb_entries:
                await rag.entities_vdb.upsert(entity_vdb_entries)

            relation_vdb_entries = self._build_relation_vectors(relations_records, file_label)
            if relation_vdb_entries:
                await rag.relationships_vdb.upsert(relation_vdb_entries)

            await asyncio.gather(
                rag.chunk_entity_relation_graph.index_done_callback(),
                rag.text_chunks.index_done_callback(),
                rag.chunks_vdb.index_done_callback(),
                rag.entities_vdb.index_done_callback(),
                rag.relationships_vdb.index_done_callback(),
            )
        finally:
            await rag.finalize_storages()

        if not processed:
            return []
        return list(dict.fromkeys(processed))

    async def _extract_entities(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not content:
            return [], []

        prompt_context = {
            "tuple_delimiter": self.tuple_delimiter,
            "completion_delimiter": self.completion_delimiter,
            "entity_types": ", ".join(self.entity_types),
            "language": self.config.graph_language,
            "examples": self.examples_block,
            "input_text": content,
        }
        system_prompt = PROMPTS["entity_extraction_system_prompt"].format(**prompt_context)
        user_prompt = PROMPTS["entity_extraction_user_prompt"].format(**prompt_context)

        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=self.config.graph_llm_max_output_tokens,
        )
        payload = response.choices[0].message.content or ""
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []

        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line or line == self.completion_delimiter:
                continue
            parts = [part.strip() for part in line.split(self.tuple_delimiter)]
            if not parts:
                continue
            tag = parts[0].lower()
            if tag == "entity" and len(parts) >= 4:
                entities.append(
                    {
                        "name": _normalize_name(parts[1]),
                        "type": _normalize_type(parts[2]),
                        "description": parts[3].strip(),
                    }
                )
            elif tag == "relation" and len(parts) >= 5:
                relations.append(
                    {
                        "source": _normalize_name(parts[1]),
                        "target": _normalize_name(parts[2]),
                        "keywords": parts[3].strip(),
                        "description": parts[4].strip(),
                    }
                )
        return entities, relations

    def _split_chunks(self, raw_content: str) -> List[Dict[str, Any]]:
        split_char = "\n\n" if self.config.graph_chunk_split_paragraphs else None
        return chunking_by_token_size(
            self.tokenizer,
            raw_content or "",
            split_by_character=split_char,
            split_by_character_only=False,
            overlap_token_size=self.config.graph_chunk_overlap,
            max_token_size=self.config.graph_chunk_size,
        )

    async def _process_chunk(
        self,
        task: ChunkTask,
        semaphore: asyncio.Semaphore,
    ) -> Optional[ChunkResult]:
        if not task.content or task.tokens < self.config.graph_min_chunk_tokens:
            return None
        try:
            async with semaphore:
                entities, relations = await self._extract_entities(task.content)
        except Exception as exc:
            logger.warning(
                "Graph builder extraction failed for %s (chunk %s): %s",
                task.url,
                task.order,
                exc,
            )
            return None

        return ChunkResult(
            source_id=task.source_id,
            url=task.url,
            chunk_id=self._chunk_id(task.url, task.order),
            order=task.order,
            tokens=task.tokens,
            content=task.content,
            entities=entities,
            relations=relations,
        )

    def _merge_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        registry: Dict[str, Dict[str, Any]],
    ) -> None:
        for entity in entities:
            name = entity.get("name")
            if not name:
                continue
            key = name.lower()
            entry = registry.get(key)
            if entry is None:
                registry[key] = {
                    "name": name,
                    "type": entity.get("type") or "Other",
                    "description": entity.get("description", ""),
                    "sources": {source_id},
                }
                continue

            if entry["type"] == "Other" and entity.get("type") and entity["type"] != "Other":
                entry["type"] = entity["type"]
            if len(entity.get("description", "")) > len(entry["description"]):
                entry["description"] = entity.get("description", "")
            entry["sources"].add(source_id)

    def _merge_relations(
        self,
        relations: List[Dict[str, Any]],
        source_id: str,
        entity_registry: Dict[str, Dict[str, Any]],
        registry: Dict[Tuple[str, str, str], Dict[str, Any]],
    ) -> None:
        for rel in relations:
            source = rel.get("source")
            target = rel.get("target")
            if not source or not target:
                continue
            if source.lower() not in entity_registry or target.lower() not in entity_registry:
                continue
            key = (source.lower(), target.lower(), rel.get("keywords", "").lower())
            entry = registry.get(key)
            if entry is None:
                registry[key] = {
                    "source": source,
                    "target": target,
                    "keywords": rel.get("keywords", ""),
                    "description": rel.get("description", ""),
                    "source_ids": {source_id},
                }
                continue

            if len(rel.get("description", "")) > len(entry["description"]):
                entry["description"] = rel.get("description", "")
            entry["source_ids"].add(source_id)

    async def _upsert_topic_node(
        self,
        rag: LightRAG,
        topic_id: str,
        topic_label: str,
        timestamp: int,
        file_label: str,
    ) -> None:
        existing = await rag.chunk_entity_relation_graph.get_node(topic_id)
        source_field = _merge_source_field(existing.get("source_id") if existing else None, {topic_id})
        description = topic_label or existing.get("description") if existing else topic_label
        await rag.chunk_entity_relation_graph.upsert_node(
            topic_id,
            node_data={
                "entity_id": topic_id,
                "entity_name": topic_label,
                "entity_type": "Topic",
                "description": description,
                "source_id": source_field,
                "file_path": file_label,
                "created_at": timestamp,
            },
        )

    async def _upsert_source_node(
        self,
        rag: LightRAG,
        topic_id: str,
        source_id: str,
        title: str,
        url: str,
        timestamp: int,
        file_label: str,
    ) -> None:
        existing = await rag.chunk_entity_relation_graph.get_node(source_id)
        source_field = _merge_source_field(existing.get("source_id") if existing else None, {source_id})
        description = title or url
        await rag.chunk_entity_relation_graph.upsert_node(
            source_id,
            node_data={
                "entity_id": source_id,
                "entity_name": title,
                "entity_type": "Source",
                "description": description,
                "source_id": source_field,
                "file_path": file_label,
                "created_at": timestamp,
                "source_url": url,
            },
        )
        existing_edge = await rag.chunk_entity_relation_graph.get_edge(topic_id, source_id)
        keywords = _merge_source_field(existing_edge.get("keywords") if existing_edge else None, {"has_source"})
        await rag.chunk_entity_relation_graph.upsert_edge(
            topic_id,
            source_id,
            edge_data={
                "keywords": keywords or "has_source",
                "description": description,
                "source_id": source_field,
                "file_path": file_label,
                "weight": 1.0,
                "created_at": timestamp,
            },
        )

    async def _upsert_entities(
        self,
        rag: LightRAG,
        entities: Dict[str, Dict[str, Any]],
        file_label: str,
        timestamp: int,
    ) -> None:
        for entry in entities.values():
            name = entry["name"]
            existing = await rag.chunk_entity_relation_graph.get_node(name)
            combined_sources = _merge_source_field(
                existing.get("source_id") if existing else None,
                entry["sources"],
            )
            description = entry["description"] or (existing.get("description") if existing else "")
            await rag.chunk_entity_relation_graph.upsert_node(
                name,
                node_data={
                    "entity_id": name,
                    "entity_name": name,
                    "entity_type": entry["type"],
                    "description": description,
                    "source_id": combined_sources,
                    "file_path": file_label,
                    "created_at": timestamp,
                },
            )
            for source_id in entry["sources"]:
                existing_edge = await rag.chunk_entity_relation_graph.get_edge(source_id, name)
                edge_keywords = _merge_source_field(
                    existing_edge.get("keywords") if existing_edge else None,
                    {"mentions"},
                )
                await rag.chunk_entity_relation_graph.upsert_edge(
                    source_id,
                    name,
                    edge_data={
                        "keywords": edge_keywords or "mentions",
                        "description": description,
                        "source_id": source_id,
                        "file_path": file_label,
                        "weight": 1.0,
                        "created_at": timestamp,
                    },
                )

    async def _upsert_relations(
        self,
        rag: LightRAG,
        relations: Dict[Tuple[str, str, str], Dict[str, Any]],
        file_label: str,
        timestamp: int,
    ) -> None:
        for entry in relations.values():
            source = entry["source"]
            target = entry["target"]
            existing = await rag.chunk_entity_relation_graph.get_edge(source, target)
            keywords = entry["keywords"] or (existing.get("keywords") if existing else "")
            description = entry["description"] or (existing.get("description") if existing else "")
            source_field = _merge_source_field(
                existing.get("source_id") if existing else None,
                entry["source_ids"],
            )
            await rag.chunk_entity_relation_graph.upsert_edge(
                source,
                target,
                edge_data={
                    "keywords": keywords,
                    "description": description,
                    "source_id": source_field,
                    "file_path": file_label,
                    "weight": 1.0,
                    "created_at": timestamp,
                },
            )

    def _build_entity_vectors(
        self,
        entities: Dict[str, Dict[str, Any]],
        file_label: str,
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for entry in entities.values():
            name = entry["name"]
            source_field = GRAPH_FIELD_SEP.join(sorted(entry["sources"]))
            entries[compute_mdhash_id(name, prefix="ent-")] = {
                "entity_name": name,
                "entity_type": entry["type"],
                "content": f"{name}\n{entry['description']}",
                "source_id": source_field,
                "file_path": file_label,
            }
        return entries

    def _build_relation_vectors(
        self,
        relations: Dict[Tuple[str, str, str], Dict[str, Any]],
        file_label: str,
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for key, entry in relations.items():
            src = entry["source"]
            tgt = entry["target"]
            source_field = GRAPH_FIELD_SEP.join(sorted(entry["source_ids"]))
            rel_id = compute_mdhash_id(f"{src}-{tgt}-{entry['keywords']}", prefix="rel-")
            entries[rel_id] = {
                "src_id": src,
                "tgt_id": tgt,
                "keywords": entry["keywords"],
                "content": f"{src}\t{tgt}\n{entry['keywords']}\n{entry['description']}",
                "source_id": source_field,
                "file_path": file_label,
                "weight": 1.0,
            }
        return entries

    @staticmethod
    async def _llm_stub(*_: Any, **__: Any) -> str:
        return ""

    @staticmethod
    def _topic_node_id(topic: str) -> str:
        return compute_mdhash_id(topic or "topic", prefix="topic-")

    @staticmethod
    def _source_node_id(url: str) -> str:
        return compute_mdhash_id(url, prefix="source-")

    @staticmethod
    def _chunk_id(url: str, index: int) -> str:
        return compute_mdhash_id(f"{url}:{index}", prefix="chunk-")


async def build_graph_workspace(
    *,
    documents: List[Dict[str, Any]],
    topic: str,
    cfg: Configuration,
    processed_urls: Iterable[str],
) -> List[str]:
    if not documents:
        return []

    builder = GraphBuilder(cfg)
    processed = await builder.build(
        topic=topic or "Research Topic",
        raw_results=documents,
        processed_urls=set(processed_urls or []),
    )
    return processed
