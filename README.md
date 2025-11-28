# Unified Memory Agent Evaluation System

A comprehensive evaluation framework for testing memory-based conversational AI agents across multiple datasets and metrics.

## Overview

This system evaluates how well AI agents can answer questions based on accumulated conversational memory over time. It supports 17 benchmark datasets including:
- **Locomo**: Multi-hop reasoning, temporal understanding, and adversarial scenarios
- **LongmemEval**: Multi-session memory, knowledge updates, and preference tracking
- **ConvoMem**: Long-context conversational memory with 6 evidence categories (107 test cases, 10K+ questions)
- **MemAlpha**: Memory-intensive multi-turn conversations
- **InfBench**: Infinite context benchmark for extremely long sequences
- **Plus 12 more**: banking77, booksum, clinic, hotpotqa, msc, nlu, perltqa, pubmed_rct, squad, synth, trec_coarse, trec_fine

## Quick Start

### Basic Usage

```bash
# Run full evaluation (generation + scoring) - async version recommended
python evaluate_async.py --task locomo --agent concat --concurrency 256

# Run with ConvoMem benchmark
python evaluate_async.py --task convomem --agent mem --concurrency 128

# Evaluate existing responses
python evaluate_async.py --task locomo --input_file results/locomo/responses_concat_[timestamp].jsonl

# Generate comprehensive statistics
python generate_stats.py --task all --save_txt

# Using convenience scripts
bash run_generation.sh      # Run generation with optimized settings
bash run_score.sh          # Run scoring on generated responses
```

### Supported Tasks & Agents

- **Tasks**: `convomem`, `locomo`, `longmemeval`, `memalpha`, `infbench`, `hotpotqa`, `msc`, `squad`, `banking77`, `booksum`, `clinic`, `nlu`, `perltqa`, `pubmed_rct`, `synth`, `trec_coarse`, `trec_fine`
- **Agents**:
  - `concat`: Concatenates all memory chunks (baseline)
  - `mem`: Advanced memory agent with structured memory management
  - `mem1`: Variant of memory agent with specialized processing
  - `emergence`: Emergence-based memory agent
  - `mem_alpha`: Optimized for MemAlpha dataset
  - `verl`: VERL (Verified Reinforcement Learning) agent

## Architecture

### Core Components

- **Agent Framework** (`/agents/`): Modular agent implementations
- **Evaluation Pipeline**: End-to-end testing workflow
- **Multi-Metric Scoring**: F1, BERT Score, ROUGE-L, Exact Match, LLM evaluation
- **Statistics Engine**: Comprehensive reporting with category breakdowns

### Evaluation Workflow

1. **Data Loading**: Process raw conversation data into structured format
2. **Memory Building**: Agents incrementally add conversation chunks
3. **Question Answering**: Questions asked at specific memory points
4. **Multi-Metric Evaluation**: Comprehensive response scoring
5. **Statistics Generation**: Detailed performance reports

## Installation

1. Clone the repository
2. Install dependencies (Python 3.8+)
3. Configure API endpoints (see Configuration section)

## Configuration

### API Setup

The system uses a custom OpenAI-compatible endpoint:
- **Base URL**: `http://api-hub.inner.chj.cloud/llm-gateway/v1`
- **Authentication**: Custom headers required
- **Models**: Azure GPT-4 variants

### Environment Variables

Set these in your environment or configuration:
```bash
export X-CHJ-GWToken="your-token-here"
export X-CHJ-GW-SOURCE="your-source-id"
```

## Usage Examples

### Run ConvoMem Evaluation
```bash
# Full pipeline with async processing
python evaluate_async.py --task convomem --agent mem --concurrency 128

# Using convenience script
bash run_generation.sh
```

### Run Locomo Evaluation
```bash
python evaluate_async.py --task locomo --agent concat --concurrency 256 --output_dir results/
```

### Run LongmemEval Evaluation
```bash
python evaluate_async.py --task longmemeval --agent mem
```

### Generate Statistics Report
```bash
# Single task
python generate_stats.py --task convomem --save_txt --output_file stats_convomem.txt

# All tasks
python generate_stats.py --task all --save_txt
```

### Batch Processing
```bash
# Run multiple benchmarks
for task in convomem locomo longmemeval memalpha; do
    python evaluate_async.py --task $task --agent mem --concurrency 256
done

# Or use the convenience script
bash run_generation.sh
bash run_score.sh
```

## Dataset Details

### ConvoMem Dataset (NEW)
- **Test Cases**: 107 long-context conversations (>2.18M characters each)
- **Questions**: 10,215 total across 6 evidence categories
- **Categories**:
  - `abstention_evidence`: 2,100 questions
  - `assistant_facts_evidence`: 2,111 questions
  - `implicit_connection_evidence`: 574 questions
  - `preference_evidence`: 4,982 questions
  - `user_evidence`: 448 questions
  - `changing_evidence`: Additional evidence type
- **Source**: Pre-mixed test cases from ConvoMem core benchmark
- **Focus**: Long-context memory retention and evidence-based question answering

### Locomo Dataset
- **Categories**: Multi-hop (1), Temporal (3), Open-domain (4), Single-hop (2), Adversarial (5)
- **Focus**: Complex reasoning and adversarial detection
- **Metrics**: Category-specific F1 scoring

### LongmemEval Dataset
- **Question Types**:
  - Multi-session memory
  - Temporal reasoning
  - Knowledge updates
  - Single-session preferences
  - Abstention scenarios
- **Evaluation**: Template-based LLM scoring

### MemAlpha Dataset
- **Focus**: Memory-intensive multi-turn conversations
- **Format**: Structured memory updates and retrieval
- **Challenges**: Long-term memory consistency

### InfBench Dataset
- **Focus**: Extremely long context understanding
- **Challenge**: Maintaining performance with infinite context
- **Use Case**: Testing context window limits

## Output Files

### Generation Phase
- **responses_*.jsonl**: Real-time JSONL output with questions and responses
- **Location**: `results/{task}/` with timestamp

### Evaluation Phase
- **evaluated_*.jsonl**: Complete evaluation results with all metrics
- **Statistics**: Pretty-printed tables (console or text file)

### File Structure
```
results/
├── locomo/
│   ├── responses_concat_[timestamp].jsonl
│   └── evaluated_concat_[timestamp].jsonl
└── longmemeval/
    ├── responses_concat_[timestamp].jsonl
    └── evaluated_concat_[timestamp].jsonl
```

## Extending the System

### Adding New Agents

1. Create new agent class inheriting from `BaseAgent`
2. Implement required methods: `add_memory()` and `QA()`
3. Add to agent factory in `evaluate.py`

Example:
```python
class CustomAgent(BaseAgent):
    def add_memory(self, chunk: str) -> None:
        # Your implementation
        pass
    
    def QA(self, question: str) -> str:
        # Your implementation
        pass
```

### Adding New Metrics

1. Add scoring method to `BaseEvaluator`
2. Update evaluator `evaluate_qa()` methods
3. Update statistics generation

### Adding New Datasets

1. Create dataset loader in `data/EvalDataset.py` using the `@register_benchmark()` decorator
2. Implement category mapping
3. Add evaluation logic
4. Update statistics generation

Example:
```python
@register_benchmark()
def load_custom_dataset(force_rebuild=False) -> list[EvalData]:
    """Load custom dataset."""
    file_path = "data/processed_custom.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)

    # Process raw data
    processed = []
    # ... your data loading logic

    json.dump([item.model_dump() for item in processed],
              open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed
```

## Available Benchmarks

The system currently supports 17 benchmarks (use `--task <benchmark_name>`):

1. **convomem** - Long-context conversational memory (107 cases, 10K+ questions)
2. **locomo** - Multi-hop reasoning and adversarial scenarios
3. **longmemeval** - Multi-session memory tracking
4. **memalpha** - Memory-intensive conversations
5. **infbench** - Infinite context benchmark
6. **hotpotqa** - Multi-hop question answering
7. **msc** - Multi-session conversations
8. **squad** - Reading comprehension
9. **banking77** - Banking intent classification
10. **booksum** - Book summarization
11. **clinic** - Clinical text understanding
12. **nlu** - Natural language understanding tasks
13. **perltqa** - Long-form QA
14. **pubmed_rct** - Medical research abstracts
15. **synth** - Synthetic conversational data
16. **trec_coarse** - TREC question classification (coarse)
17. **trec_fine** - TREC question classification (fine)

## Troubleshooting

### Common Issues

**API Rate Limits**: The system includes built-in retry logic with exponential backoff.

**Memory Issues**: For large datasets, consider batch processing or memory optimization.

**Missing Dependencies**: Ensure all Python packages are installed:
```bash
pip install -r requirements.txt  # if requirements.txt exists
```

### Debug Mode

Enable verbose logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

- Use batch processing for large datasets
- Monitor API rate limits
- Consider parallel processing for multiple tasks
- Use SSD storage for better I/O performance

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request
