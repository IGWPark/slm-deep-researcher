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

See [`notebooks/research_example.ipynb`](notebooks/research_example.ipynb) for complete implementation.

The agent generates queries dynamically and refines its search through multiple iterations:

```
🔍 Initial Query
   environmental impact AI model training comparison 2025

✓ Found 3 sources
   The Carbon Footprint of Machine Learning Training...
   Energy and Policy Considerations for Deep Learning in NLP
   Quantifying the Carbon Emissions of Machine Learning

📝 Summary updated

🔍 Follow-up Query #1
   energy consumption small models vs large models

✓ Found 3 sources
   Small Language Models: Survey, Measurements, and Insights
   Efficient Large Language Models: A Survey

📝 Summary updated

🔍 Follow-up Query #2
   carbon footprint distributed training thousands of models

✓ Found 3 sources
📝 Summary updated
```

The agent reflects on each summary to identify knowledge gaps, then generates new queries to fill those gaps.

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
