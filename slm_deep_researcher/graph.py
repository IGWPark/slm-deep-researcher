"""The main research workflow that coordinates query generation, search, and summarization."""

from __future__ import annotations

import json
from typing import Any, Dict
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Configuration
from .llm import create_chat_model
from .prompts import (
    build_reflection_prompt,
    get_current_year,
    json_mode_query_instructions,
    json_mode_reflection_instructions,
    query_writer_instructions,
    summarizer_instructions,
    tool_calling_query_instructions,
    tool_calling_reflection_instructions,
)
from .state import ResearchInput, ResearchOutput, ResearchState
from .utils import (
    deduplicate_and_format_sources,
    duckduckgo_search,
    format_sources,
    strip_thinking_tokens,
)
from .token import truncate_to_fit


def _invoke_tool_mode(config: Configuration, messages: list, tool_cls, max_tokens: int):
    llm = create_chat_model(config, max_tokens=max_tokens)
    return llm.bind_tools([tool_cls]).invoke(messages)


def _invoke_json_mode(config: Configuration, messages: list, max_tokens: int):
    llm = create_chat_model(config, json_mode=True, max_tokens=max_tokens)
    return llm.invoke(messages)


def _to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return str(content)


def _parse_structured_output(content: str, field: str) -> Any:
    try:
        payload = json.loads(content)
        return payload.get(field)
    except json.JSONDecodeError:
        return None


def generate_query(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    fallback_query = f"Tell me more about {state.research_topic}"
    region = state.ddgs_region or cfg.ddgs_region

    @tool
    class QueryTool(BaseModel):
        query: str = Field(description="The actual search query string")
        rationale: str = Field(description="Brief explanation of why this query is relevant")

    instruction = (
        tool_calling_query_instructions
        if cfg.use_tool_calling == "tools"
        else json_mode_query_instructions
    )
    messages = [
        SystemMessage(
            content=query_writer_instructions.format(
                current_year=get_current_year(), research_topic=state.research_topic
            )
            + instruction
        ),
        HumanMessage(content="Generate a query for web search:"),
    ]

    if cfg.use_tool_calling == "tools":
        response = _invoke_tool_mode(cfg, messages, QueryTool, cfg.llm_max_output_tokens_query)
        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            search_query = tool_calls[0].get("args", {}).get("query") or fallback_query
        else:
            search_query = fallback_query
    else:
        response = _invoke_json_mode(cfg, messages, cfg.llm_max_output_tokens_query)
        content = strip_thinking_tokens(_to_str(response.content)) if cfg.llm_strip_thinking else _to_str(response.content)
        search_query = _parse_structured_output(content, "query") or fallback_query

    return {
        "search_query": search_query,
        "search_query_history": [search_query],
        "ddgs_region": region,
    }


def web_research(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    search_results = duckduckgo_search(
        state.search_query,
        max_results=cfg.search_max_results,
        fetch_full_page=cfg.fetch_full_page,
        region=state.ddgs_region or cfg.ddgs_region,
        sites=cfg.search_sites,
        inurls=cfg.search_inurl,
    )
    formatted_sources = deduplicate_and_format_sources(
        search_results,
        max_tokens_per_source=cfg.max_tokens_per_source,
        fetch_full_page=cfg.fetch_full_page,
    )
    return {
        "search_query_history": [state.search_query],
        "sources_gathered": [format_sources(search_results)],
        "web_research_results": [formatted_sources],
        "research_loop_count": state.research_loop_count + 1,
    }


def summarize_sources(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    llm = create_chat_model(cfg, max_tokens=cfg.llm_max_output_tokens_summary)

    if state.running_summary:
        human_message_content = (
            f"<Existing Summary> \n {state.running_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {state.web_research_results[-1]} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {state.web_research_results[-1]} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )

    response = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    summary = strip_thinking_tokens(_to_str(response.content)) if cfg.llm_strip_thinking else _to_str(response.content)
    return {"running_summary": summary}


def reflect_on_summary(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)

    @tool
    class FollowUpQuery(BaseModel):
        follow_up_query: str = Field(description="Write a specific question to address this gap")
        knowledge_gap: str = Field(description="Describe what information is missing or needs clarification")

    prompt = build_reflection_prompt(
        research_topic=state.research_topic,
        previous_queries=state.search_query_history if state.search_query_history else None
    )

    instruction = (
        tool_calling_reflection_instructions
        if cfg.use_tool_calling == "tools"
        else json_mode_reflection_instructions
    )
    system_content = prompt + instruction
    user_content = state.running_summary or "Summaries missing; synthesize actionable follow-ups."

    system_content, user_content = truncate_to_fit(
        system_content,
        user_content,
        cfg.llm_max_input_tokens,
        cfg.llm_max_output_tokens_reflection,
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    if cfg.use_tool_calling == "tools":
        response = _invoke_tool_mode(cfg, messages, FollowUpQuery, cfg.llm_max_output_tokens_reflection)
        tool_calls = getattr(response, "tool_calls", [])
        follow_up_query = tool_calls[0].get("args", {}).get("follow_up_query") if tool_calls else None
    else:
        response = _invoke_json_mode(cfg, messages, cfg.llm_max_output_tokens_reflection)
        content = strip_thinking_tokens(_to_str(response.content)) if cfg.llm_strip_thinking else _to_str(response.content)
        follow_up_query = _parse_structured_output(content, "follow_up_query")

    # If reflection fails, generate a generic follow-up instead of repeating queries
    fallback_query = f"Additional research on {state.research_topic}"
    return {"search_query": follow_up_query or fallback_query}


def finalize_summary(state: ResearchState) -> Dict[str, Any]:
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        for line in source.split("\n"):
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    all_sources = "\n".join(unique_sources)
    final_summary = f"## Summary\n{state.running_summary}\n\n### Sources:\n{all_sources}"
    return {"running_summary": final_summary}


def route_research(state: ResearchState, config: RunnableConfig) -> Literal["web_research", "finalize_summary"]:
    cfg = Configuration.from_runnable_config(config)
    if state.research_loop_count <= cfg.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState, input_schema=ResearchInput, output_schema=ResearchOutput)
    graph.add_node("generate_query", generate_query)
    graph.add_node("web_research", web_research)
    graph.add_node("summarize_sources", summarize_sources)
    graph.add_node("reflect_on_summary", reflect_on_summary)
    graph.add_node("finalize_summary", finalize_summary)

    graph.add_edge(START, "generate_query")
    graph.add_edge("generate_query", "web_research")
    graph.add_edge("web_research", "summarize_sources")
    graph.add_edge("summarize_sources", "reflect_on_summary")
    graph.add_conditional_edges("reflect_on_summary", route_research)
    graph.add_edge("finalize_summary", END)

    return graph
