# Unified Memory Agent Evaluation System

A comprehensive evaluation framework for testing memory-based conversational AI agents across multiple datasets and metrics.

## Overview

This system evaluates how well AI agents can answer questions based on accumulated conversational memory over time. It supports two main datasets:
- **Locomo**: Focuses on multi-hop reasoning, temporal understanding, and adversarial scenarios
- **LongmemEval**: Tests multi-session memory, knowledge updates, and preference tracking

## Quick Start

### Basic Usage

```bash
# Run full evaluation (generation + evaluation)
python evaluate.py --task locomo --agent concat

# Evaluate existing responses
python evaluate.py --task locomo --input_file results/locomo/responses_concat_[timestamp].jsonl

# Async version
python evaluate_async.py --task longmemeval --agent concat --concurrency 256

# Generate comprehensive statistics
python generate_stats.py --task all --save_txt
```

### Supported Tasks & Agents

- **Tasks**: `locomo`, `longmemeval`
- **Agents**: `concat` (concatenates all memory chunks)

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

### Run Locomo Evaluation
```bash
python evaluate.py --task locomo --agent concat --output_dir results/
```

### Run LongmemEval Evaluation
```bash
python evaluate.py --task longmemeval --agent concat
```

### Generate Statistics Report
```bash
python generate_stats.py --task locomo --save_txt --output_file stats_report.txt
```

### Batch Processing
```bash
# Run both tasks sequentially
python evaluate.py --task locomo --agent concat && python evaluate.py --task longmemeval --agent concat
```

## Dataset Details

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

1. Create dataset loader following `EvalDataset.py` pattern
2. Implement category mapping
3. Add evaluation logic
4. Update statistics generation

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

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue on the repository or contact the development team.