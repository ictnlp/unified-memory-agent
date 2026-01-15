# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Unified Memory Agent** evaluation system for testing memory-based conversational AI agents. The system evaluates how well agents can answer questions based on accumulated conversational memory over time using two main datasets: Locomo and LongmemEval.

## Core Architecture

### Agent Framework (`/agents/`)
- **BaseAgent**: Abstract base class defining the agent interface with OpenAI client configuration
- **ConcatAgent**: Primary implementation that concatenates all memory chunks for Q&A
- **Factory Pattern**: Use `get_agent(agent_name, **kwargs)` to instantiate agents

### Evaluation Pipeline
The system follows this workflow:
1. **Data Loading**: `EvalDataset.py` processes raw conversation data into structured `EvalData` objects
2. **Memory Building**: Agents incrementally add conversation chunks as memory
3. **Question Answering**: Questions are asked at specific memory accumulation points
4. **Multi-Metric Evaluation**: Responses evaluated using F1, BERT Score, ROUGE-L, Exact Match, and LLM scoring
5. **Statistics Generation**: Comprehensive reporting with category breakdowns

### Key Components
- **evaluate.py**: Main evaluation script with generation and evaluation logic
- **generate_stats.py**: Statistics generation with prettytable output  
- **utils.py**: Text normalization and F1 scoring utilities
- **data/EvalDataset.py**: Dataset loading and preprocessing

## Development Commands

### Running Evaluations
```bash
# Generate responses and evaluate
python evaluate.py --task locomo --agent concat

# Evaluate existing responses  
python evaluate.py --task locomo --input_file results/locomo/responses_concat_*.jsonl

# Generate comprehensive statistics
python generate_stats.py --task all --save_txt
```

### Supported Parameters
- **Tasks**: `locomo`, `longmemeval`
- **Agents**: `concat` (currently the only implementation)
- **Evaluation Modes**: Generation + evaluation, or evaluation-only with `--input_file`

## API Configuration

The system uses OpenAI-compatible endpoints. Configure your API settings in the `.env` file:
- **Base URL**: Set via `REMOTE_API_BASE` environment variable
- **Authentication**: Custom headers (X-CHJ-GWToken, X-CHJ-GW-SOURCE) set via environment variables
- **Models**: Azure GPT-4 variants via MODEL_NAME_MAP

## Dataset Structure

### Locomo Dataset
- **Categories**: Multi-hop (1), Temporal (3), Open-domain (4), Single-hop (2), Adversarial (5)
- **Evaluation**: Category-specific F1 scoring logic with adversarial detection

### LongmemEval Dataset  
- **Question Types**: Multi-session, temporal-reasoning, knowledge-update, single-session-preference, abstention
- **Evaluation**: Template-based LLM scoring with question-type-specific prompts

## Memory Processing Logic

**Critical Timing**: Questions are processed at their designated memory position BEFORE adding new memory chunks to avoid future information leakage.

```python
# Correct order in generation loop:
questions_at_position = get_questions_for_position(current_position)
responses = agent.QA_batch(questions_at_position)  # Process with current memory
agent.add_memory(new_chunk)  # Add memory after processing
```

## Output Files

### Generation Phase
- **responses_*.jsonl**: Real-time JSONL output with qid, query, response, expected_answer
- Files saved to `results/{task}/` with timestamp

### Evaluation Phase  
- **evaluated_*.jsonl**: Evaluation results with all metrics
- **Category mapping**: Uses original dataset files for category information

### Statistics
- **Console output**: Pretty-printed tables with overall and category-specific performance
- **Optional text files**: Use `--save_txt` flag

## Key Implementation Notes

- **Batch Processing**: Agents support `QA_batch()` for efficient multiple question handling
- **Error Handling**: Robust retry logic with backoff for API calls
- **Memory Management**: Incremental memory building as conversations progress
- **Multi-Metric Evaluation**: BaseEvaluator provides standardized scoring methods across tasks

## Extending the System

### Adding New Agents
1. Inherit from `BaseAgent`
2. Implement `add_memory()` and `QA()` methods
3. Add to agent factory in `evaluate.py`

### Adding New Metrics
1. Add scoring method to `BaseEvaluator`
2. Update evaluator `evaluate_qa()` methods to include new metric
3. Update statistics generation accordingly