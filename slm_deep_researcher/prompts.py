"""Prompts optimized for small models with limited context."""

from datetime import datetime


def get_current_date() -> str:
    return datetime.now().strftime("%B %d, %Y")


def get_current_year() -> str:
    return datetime.now().strftime("%Y")


query_writer_instructions = """Generate a web search query for this topic.

<TOPIC>
{research_topic}
</TOPIC>

<RULES>
- 3-8 words maximum
- Keyword format (not a question)
- Include year {current_year} if topic is time-sensitive
- Use specific terms, avoid filler words
</RULES>"""

json_mode_query_instructions = """<FORMAT>
Respond with valid JSON only:
{{"query": "your search query here", "rationale": "why this query"}}
</FORMAT>"""

tool_calling_query_instructions = """<INSTRUCTIONS>
Call the Query tool with:
- query: search query string
- rationale: why this query
</INSTRUCTIONS>"""

summarizer_instructions = """You are a research summarizer. Create clear, informative summaries.

<TASK>
NEW summary: Extract and organize key information from search results related to the topic.
UPDATE summary: Integrate new information into existing summary. Merge related points, add new insights.
</TASK>

<RULES>
- Focus on factual information
- Maintain logical flow
- Skip redundant information
- No preamble or titles
- Start directly with content
</RULES>"""

reflection_instructions = """Analyze this summary and identify missing information.

<TOPIC>
{research_topic}
</TOPIC>

<GOAL>
Generate a web search query to fill knowledge gaps.
Focus on: details, implementations, recent developments, or comparisons.
</GOAL>

<RULES>
- 3-10 words maximum
- Keyword format (not a question)
- Explore a different aspect than previous queries{previous_queries_constraint}
</RULES>"""

def build_reflection_prompt(research_topic: str, previous_queries: list[str] | None = None) -> str:
    """Builds reflection prompt and includes previous queries to avoid repetition."""
    if previous_queries and len(previous_queries) > 0:
        queries_text = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(previous_queries))
        constraint = f"\n- DO NOT repeat these previous queries:\n{queries_text}\n- Generate a DIFFERENT follow-up exploring a NEW aspect"
    else:
        constraint = ""

    return reflection_instructions.format(
        research_topic=research_topic,
        previous_queries_constraint=constraint
    )

json_mode_reflection_instructions = """<FORMAT>
Respond with valid JSON only:
{{"knowledge_gap": "what's missing", "follow_up_query": "specific question"}}
</FORMAT>"""

tool_calling_reflection_instructions = """<INSTRUCTIONS>
Call the FollowUpQuery tool with:
- follow_up_query: specific question
- knowledge_gap: what's missing
</INSTRUCTIONS>"""
