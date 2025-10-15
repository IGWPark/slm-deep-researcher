"""State that gets passed between workflow nodes."""

from dataclasses import dataclass, field
import operator
from typing_extensions import Annotated


@dataclass(kw_only=True)
class ResearchState:
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    search_query_history: Annotated[list, operator.add] = field(default_factory=list)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    web_research_documents: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)
    ddgs_region: str = field(default=None)
    graph_indexed_urls: Annotated[list, operator.add] = field(default_factory=list)


@dataclass(kw_only=True)
class ResearchInput:
    research_topic: str = field(default=None)


@dataclass(kw_only=True)
class ResearchOutput:
    running_summary: str = field(default=None)
