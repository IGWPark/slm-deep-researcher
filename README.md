<div align="center">

# SLM Deep Researcher

**Web research agent for small language models**

</div>

---

A variation of [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher) designed to test small language model capabilities under resource constraints. Uses only DuckDuckGo for search (no paid APIs) and runs on models as small as 1.7B parameters.

## About the Model

This project uses [Menlo/Lucy](https://huggingface.co/Menlo/Lucy), a 1.7B parameter model based on Qwen3-1.7B. Lucy is specifically designed for agentic search tasks and generates reasoning tokens (`<think>` blocks) before responses, making it suitable for understanding how small models handle multi-step research workflows.

## Changes for Smaller Models

To make this work with small language models, the following modifications were made:

**Prompts**: Query generation and reflection prompts use keyword format instead of natural language questions. Queries are limited to 3-8 words for initial searches and 3-10 words for follow-ups to reduce token usage.

**Token Allocation**: Output limits are set to 512 tokens for query/reflection and 2048 for summaries. This accounts for reasoning models that generate thinking tokens before tool calls. Non-reasoning models may work with lower limits.

**Search API**: Uses DuckDuckGo exclusively (no paid APIs required).

**Provider Support**: Compatible with any OpenAI-compatible server (vLLM, Ollama, etc).

**Graph Capability**: Optional collector service extracts entities/relations with the same SLM and merges them into a LightRAG-compatible graph (graphML + NanoVectorDB JSON), enabling downstream graph-based retrieval or visualization.

## Tested Environment

- Model: [Menlo/Lucy-1.7B-v1.0](https://huggingface.co/Menlo/Lucy) (Qwen3-1.7B base)
- Server: vLLM
- Context window: 16384 tokens
- Features: Tool calling, reasoning tokens

## Requirements

- Python 3.12+
- LLM server (vLLM, Ollama, or OpenAI-compatible)
- LangGraph CLI

## Installation

```bash
git clone https://github.com/IGWPark/slm-deep-researcher.git
cd slm-deep-researcher
pip install -r requirements.txt
```

## Setup vLLM Server

Install vLLM:

```bash
pip install vllm
```

Start the server with Menlo/Lucy:

```bash
vllm serve Menlo/Lucy \
  --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

The server will be available at `http://localhost:8000/v1`

## Configuration

Copy the example configuration:

```bash
cp .env.example .env
```

Key settings:

```env
LLM_MODEL=your-model-name                      # Leave empty to auto-detect
LLM_BASE_URL=http://localhost:8000/v1          # vLLM defaults to 8000, Ollama uses 11434
LLM_TEMPERATURE=0.7                            # Recommended for Lucy
LLM_MAX_OUTPUT_TOKENS_QUERY=512                # Query generation budget
LLM_MAX_OUTPUT_TOKENS_REFLECTION=512           # Follow-up query budget
LLM_MAX_OUTPUT_TOKENS_SUMMARY=2048             # Summary budget

MAX_WEB_RESEARCH_LOOPS=3                       # Number of research rounds
SEARCH_MAX_RESULTS=3                           # Results per query
FETCH_FULL_PAGE=True                           # Extract full page content
USE_TOOL_CALLING=tools                         # "tools" or "json" mode
DDGS_REGION=us-en                              # DuckDuckGo region
```

## Usage

Start the LangGraph server:

```bash
langgraph dev --config langgraph.json --no-browser
```

The server runs at `http://localhost:2024` by default.

## Example

### Multi-Source Research

Run parallel research agents with different source filters:

```bash
python examples/multi_source_research.py
```

Demonstrates running multiple agents concurrently, each targeting different sources, with results saved as markdown reports.

### Graph-Enabled Research

We treat entity/relation extraction as a first-class capability for small models: the collector service buffers raw sources, calls the SLM to extract structured knowledge, and builds a LightRAG-compatible graph (graphML + NanoVectorDB JSON). That graph can be visualized with NetworkX or served via LightRAG.

1. Install optional graph dependencies (collector API + LightRAG runtime):

   ```bash
   pip install -r requirements-graph.txt
   ```

2. Launch the collector service (separate terminal):
   ```bash
   vllm serve Qwen/Qwen3-Embedding-0.6B \
     --task embed \
     --port 8001
   ```

   ```bash
   python -m slm_deep_researcher.graph_collector
   ```

3. Run the wiki/blog/news example with the collector enabled so all sources merge into one graph:

   ```bash
   GRAPH_COLLECTOR_URL=http://127.0.0.1:8085 python examples/graph_enabled_research.py
   ```

   Reports land in `reports/`, and the combined LightRAG workspace is written to `graph_outputs_server/<timestamp>/`.

4. Inspect or serve the graph:

   ```bash
   lightrag-server --workspace graph_outputs_server/<timestamp>
   ```

   (You can also load the GraphML into NetworkX or other graph tooling.)

If you skip `GRAPH_COLLECTOR_URL`, the script falls back to building the graph locally after each run.

### API Usage

Using the LangGraph API with configurable parameters:

```python
import requests

response = requests.post(
    "http://localhost:2024/runs/stream",
    json={
        "assistant_id": "your_assistant_id",
        "input": {"research_topic": "your research topic"},
        "config": {
            "configurable": {
                "search_sites": "en.wikipedia.org",
                "search_inurl": "wiki"
            }
        }
    }
)
```

### Filtering Options

Target specific domains:
```python
config = {"configurable": {"search_sites": ".edu,en.wikipedia.org"}}
```

Filter by URL patterns:
```python
config = {"configurable": {"search_inurl": "news,blog"}}
```

Combine both filters:
```python
config = {
    "configurable": {
        "search_sites": ".edu,.gov",
        "search_inurl": "research"
    }
}
```

The agent generates queries dynamically and refines its search through multiple iterations, reflecting on each summary to identify knowledge gaps and generate follow-up queries.

## Architecture

The workflow follows [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)'s design:

1. **generate_query** - Creates initial search query
2. **web_research** - Searches and extracts content
3. **summarize_sources** - Builds running summary
4. **reflect_on_summary** - Identifies missing information, generates follow-ups
5. **Loop** - Repeats steps 2-4 until reaching max loops
6. **finalize_summary** - Formats final output with sources

## Project Structure

```
slm_deep_researcher/
├── graph.py          # LangGraph workflow
├── prompts.py        # Prompt templates
├── llm.py           # LLM client setup
├── utils.py         # Search and formatting
├── state.py         # State definitions
├── config.py        # Configuration
└── token.py         # Token management
```

## Model Requirements

Required features:
- Tool calling or JSON mode support
- Minimum 512 tokens for query/reflection outputs

Lucy generates `<think>` tokens (150-200 tokens) before responding. Non-reasoning models may work with lower token limits.
