#!/usr/bin/env python3
"""Generate evaluation statistics tables from evaluation outputs."""

import os
import json
import glob
import argparse
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import sys
sys.path.append('.')

from config import DATASET_LOADERS


def _extract_agent_name(filename: str, prefix: str) -> str:
    """Extract the agent name from filenames like prefix_agent[_timestamp].jsonl."""
    base = filename
    if base.endswith('.jsonl'):
        base = base[:-6]
    if base.startswith(prefix):
        base = base[len(prefix):]
    parts = base.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return base

def parse_args():
    parser = argparse.ArgumentParser(description='Generate evaluation statistics')
    parser.add_argument('--results_dir', type=str, default='results/qwen3-4b',
                        help='Results directory containing task subdirectories')
    parser.add_argument('--task', type=str, default='all',
                        help='Task to generate statistics for (default: all subdirectories)')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save statistics to txt files (default: False)')
    return parser.parse_args()


def load_agent_ignore(results_dir: str) -> tuple[set[str], Optional[str]]:
    """Load ignored agents from .agentignore located in results_dir or its ancestors."""
    current = os.path.abspath(results_dir)
    while True:
        ignore_file = os.path.join(current, '.agentignore')
        if os.path.isfile(ignore_file):
            with open(ignore_file, 'r', encoding='utf-8') as f:
                entries = {line.strip() for line in f if line.strip() and not line.startswith('#')}
            return entries, ignore_file
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return set(), None


def load_bench_ignore(results_dir: str) -> tuple[set[str], Optional[str]]:
    """Load ignored benchmarks from .benchignore located in results_dir or its ancestors."""
    current = os.path.abspath(results_dir)
    while True:
        ignore_file = os.path.join(current, '.benchignore')
        if os.path.isfile(ignore_file):
            with open(ignore_file, 'r', encoding='utf-8') as f:
                entries = {line.strip() for line in f if line.strip() and not line.startswith('#')}
            return entries, ignore_file
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return set(), None


def discover_tasks(results_dir: str, requested_task: str) -> List[str]:
    """Return the list of tasks to process."""
    if requested_task != 'all':
        return [requested_task]

    if not os.path.isdir(results_dir):
        return []

    tasks = [
        name for name in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, name)) and not name.startswith('.')
    ]
    return tasks

def load_evaluation_results(results_dir, task):
    """Load all evaluation results for a task"""
    task_dir = os.path.join(results_dir, task)
    if not os.path.exists(task_dir):
        print(f"Task directory {task_dir} not found")
        return {}
    
    # Find all evaluated_*.jsonl files
    eval_files = glob.glob(os.path.join(task_dir, "evaluated_*.jsonl"))
    
    agent_results = defaultdict(list)
    
    for file_path in eval_files:
        filename = os.path.basename(file_path)
        agent_name = _extract_agent_name(filename, 'evaluated_')

        # Load results
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                agent_results[agent_name].append(result)
    
    return agent_results


def discover_response_agents(results_dir: str, task: str) -> Dict[str, bool]:
    """Return a mapping of agent names to True if a responses_ file exists."""
    task_dir = os.path.join(results_dir, task)
    if not os.path.exists(task_dir):
        return {}

    responses = {}
    response_files = glob.glob(os.path.join(task_dir, 'responses_*.jsonl'))
    for file_path in response_files:
        filename = os.path.basename(file_path)
        agent_name = _extract_agent_name(filename, 'responses_')
        responses[agent_name] = True
    return responses

def calculate_task_statistics(agent_results, task):
    """Calculate statistics for a task"""
    stats = {}
    
    # Build qid to category mapping from original data
    qid_to_category = build_qid_category_mapping(task)
    
    for agent_name, results in agent_results.items():
        if not results:
            continue
            
        agent_stats = {
            'total_questions': len(results),
            'agent': agent_name
        }
        
        # Extract all metrics
        all_metrics = defaultdict(list)
        category_stats = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            metrics = result.get('metric', {})
            qid = result.get('qid', '')
            
            # Get category from original data
            category = qid_to_category.get(qid, 'unknown')
            
            # Skip abstention_evidence category for convomem task
            if task == 'convomem' and category != 'user_evidence':
                continue
            # if task == 'locomo' and category == 5:
            #     continue
            
            # Collect all numeric metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
                    # Also add to category stats
                    category_stats[category][metric_name].append(value)
        
        # Calculate averages for overall metrics
        for metric_name, values in all_metrics.items():
            # If any value is -1, set average to -1
            if any(v == -1 for v in values):
                agent_stats[f'{metric_name}_avg'] = -1
                agent_stats[f'{metric_name}_total'] = -1
            else:
                agent_stats[f'{metric_name}_avg'] = sum(values) / len(values) if values else 0
                agent_stats[f'{metric_name}_total'] = sum(values)
        
        # Calculate category-specific averages
        agent_stats['categories'] = {}
        for category, category_metrics in category_stats.items():
            agent_stats['categories'][category] = {}
            for metric_name, values in category_metrics.items():
                # If any value is -1, set average to -1
                if any(v == -1 for v in values):
                    agent_stats['categories'][category][f'{metric_name}_avg'] = -1
                else:
                    agent_stats['categories'][category][f'{metric_name}_avg'] = sum(values) / len(values) if values else 0
                agent_stats['categories'][category][f'{metric_name}_count'] = len(values)
        
        stats[agent_name] = agent_stats
    
    return stats

def build_qid_category_mapping(task):
    """Build mapping from qid to category using dataset loaders when available."""
    loader = DATASET_LOADERS.get(task)
    if loader is None:
        return {}

    qid_to_category = {}
    try:
        eval_set = loader()
    except Exception as exc:  # pragma: no cover - depends on dataset availability
        print(f"Warning: failed to load dataset for {task}: {exc}")
        return qid_to_category

    for sample in eval_set:
        for question in getattr(sample, 'questions', []):
            if getattr(question, 'qid', None) and getattr(question, 'category', None) is not None:
                qid_to_category[question.qid] = question.category

    return qid_to_category

def generate_overall_table(stats, task):
    """Generate overall performance table"""
    if not stats:
        return None

    table = PrettyTable()
    table.title = f"{task.upper()} - Overall Performance"

    # Determine columns based on available metrics
    agents = sorted(stats.keys())  # Sort agent names alphabetically
    if not agents:
        return None

    # Collect all unique metric columns from all agents
    all_metric_columns = set()
    for agent_stats in stats.values():
        for col in agent_stats.keys():
            if col.endswith('_avg') and 'categories' not in col:
                all_metric_columns.add(col)

    # Sort metric columns for consistent display
    metric_columns = sorted(all_metric_columns)

    columns = ['Agent', 'Total Questions'] + [col.replace('_avg', '').upper() for col in metric_columns]
    table.field_names = columns

    for agent_name in agents:  # Iterate in sorted order
        agent_stats = stats[agent_name]
        row = [agent_name, agent_stats['total_questions']]
        for metric_col in metric_columns:
            value = agent_stats.get(metric_col, -1)
            row.append(f"{value:.4f}")
        table.add_row(row)

    return table

def generate_category_table(stats, task):
    """Generate category-specific performance table"""
    if not stats:
        return None

    # Collect all categories
    all_categories = set()
    for agent_stats in stats.values():
        all_categories.update(agent_stats.get('categories', {}).keys())

    if not all_categories:
        return None

    table = PrettyTable()
    table.title = f"{task.upper()} - Performance by Category"

    # Collect all unique metric types from all agents and categories
    all_metric_types = set()
    for agent_stats in stats.values():
        for category_stats in agent_stats.get('categories', {}).values():
            for col in category_stats.keys():
                if col.endswith('_avg'):
                    all_metric_types.add(col.replace('_avg', ''))

    # Sort metric types for consistent display
    metric_types = sorted(all_metric_types)

    if not metric_types:
        return None

    # Create table with dynamic columns
    columns = ['Agent', 'Category', 'Count'] + [m.upper() for m in metric_types]
    table.field_names = columns

    # Group by category first, then by agent for easier comparison
    for category in sorted(all_categories):
        for agent_name in sorted(stats.keys()):  # Sort agent names
            categories = stats[agent_name].get('categories', {})
            if category in categories:
                cat_stats = categories[category]
                # Use first metric type to get count (all should have same count)
                count = cat_stats.get(f'{metric_types[0]}_count', 0) if metric_types else 0
                row = [agent_name, str(category), count]
                for metric in metric_types:
                    value = cat_stats.get(f'{metric}_avg', -1)
                    row.append(f"{value:.4f}")
                table.add_row(row)

    return table

def save_tables_to_file(tables, task, results_dir):
    """Save per-task tables to file."""
    output_file = os.path.join(results_dir, task, f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Statistics for {task.upper()}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for table in tables:
            if table:
                f.write(str(table) + "\n\n")

    print(f"Statistics saved to: {output_file}")


def save_summary_table(summary_table: PrettyTable, results_dir: str):
    """Persist the cross-task summary table."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f"summary_llm_scores_{timestamp}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"LLM Score Summary\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(summary_table) + "\n")
    print(f"Summary table saved to: {output_file}")


def generate_summary_table(summary_scores: Dict[str, Dict[str, float]], tasks: List[str]) -> Optional[PrettyTable]:
    """Create a table with agents as rows and tasks as columns for llm_score."""
    if not summary_scores:
        return None

    sorted_tasks = sorted(tasks)
    table = PrettyTable()
    table.title = "LLM Score Summary"
    table.field_names = ['Agent'] + [task.upper() for task in sorted_tasks] + ['AVG']

    # Find max value for each task (column) excluding -1 and None
    max_values_by_task = {}
    for task in sorted_tasks:
        valid_values = [
            summary_scores[agent].get(task)
            for agent in summary_scores.keys()
            if summary_scores[agent].get(task) is not None and summary_scores[agent].get(task) != -1
        ]
        max_values_by_task[task] = max(valid_values) if valid_values else None

    # Calculate max average across all agents
    agent_averages = {}
    for agent in summary_scores.keys():
        valid_scores = [
            score for score in summary_scores[agent].values()
            if score is not None and score != -1
        ]
        if valid_scores:
            agent_averages[agent] = sum(valid_scores) / len(valid_scores)
        else:
            agent_averages[agent] = None
    
    max_avg = max([avg for avg in agent_averages.values() if avg is not None], default=None)

    for agent in sorted(summary_scores.keys()):
        row = [agent]
        for task in sorted_tasks:
            value = summary_scores[agent].get(task)
            if value is None or value == -1:
                row.append('-')
            elif max_values_by_task[task] is not None and abs(value - max_values_by_task[task]) < 1e-6:
                # Bold green for SOTA value (bright and eye-catching)
                row.append(f"\033[1;32m{value:.4f}\033[0m")
            else:
                row.append(f"{value:.4f}")
        
        # Add average column
        avg_value = agent_averages[agent]
        if avg_value is None:
            row.append('-')
        elif max_avg is not None and abs(avg_value - max_avg) < 1e-6:
            # Bold green for best average
            row.append(f"\033[1;32m{avg_value:.4f}\033[0m")
        else:
            row.append(f"{avg_value:.4f}")
        
        table.add_row(row)

    return table


def generate_response_presence_table(presence: Dict[str, Dict[str, bool]], tasks: List[str]) -> Optional[PrettyTable]:
    """Create a table marking which agents have response files for each task."""
    if not presence:
        return None

    sorted_tasks = sorted(tasks)
    table = PrettyTable()
    table.title = "Response File Availability"
    table.field_names = ['Agent'] + [task.upper() for task in sorted_tasks]

    for agent in sorted(presence.keys()):
        row = [agent]
        for task in sorted_tasks:
            row.append('✓' if presence[agent].get(task) else '-')
        table.add_row(row)

    return table


def save_response_table(response_table: PrettyTable, results_dir: str):
    """Persist the response availability table."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f"summary_responses_{timestamp}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Response File Availability\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(response_table) + "\n")
    print(f"Response availability table saved to: {output_file}")


def extract_session_count(task_name: str) -> Optional[int]:
    """Extract session count from task names like synth-ss4, synth-ss10, etc."""
    match = re.match(r'synth-ss(\d+)', task_name)
    if match:
        return int(match.group(1))
    return None


def plot_session_scaling(summary_scores: Dict[str, Dict[str, float]],
                         tasks: List[str],
                         results_dir: str):
    """Plot performance curves for synth-ss benchmarks with different session counts."""

    # Filter synth-ss tasks and extract session counts
    synth_tasks = []
    for task in tasks:
        session_count = extract_session_count(task)
        if session_count is not None:
            synth_tasks.append((task, session_count))

    if not synth_tasks:
        print("\nNo synth-ss benchmarks found for plotting.")
        return

    # Sort by session count
    synth_tasks.sort(key=lambda x: x[1])

    # Prepare data for plotting
    agent_data = defaultdict(lambda: {'sessions': [], 'scores': []})

    for task, session_count in synth_tasks:
        for agent_name, scores in summary_scores.items():
            if task in scores and scores[task] is not None and scores[task] != -1:
                agent_data[agent_name]['sessions'].append(session_count)
                agent_data[agent_name]['scores'].append(scores[task])

    if not agent_data:
        print("\nNo valid data for synth-ss benchmarks plotting.")
        return

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot each agent's curve
    for agent_name in sorted(agent_data.keys()):
        data = agent_data[agent_name]
        if data['sessions']:
            plt.plot(data['sessions'], data['scores'],
                    marker='o', linewidth=2, markersize=8,
                    label=agent_name)

    plt.xlabel('Number of Sessions', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Agent Performance vs Session Count (synth-ss benchmarks)',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f"synth_session_scaling_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSession scaling plot saved to: {output_file}")
    plt.close()


def main():
    args = parse_args()

    tasks = discover_tasks(args.results_dir, args.task)
    if not tasks:
        print("No tasks found. Please check the results directory or --task argument.")
        return

    ignored_benchmarks, bench_ignore_path = load_bench_ignore(args.results_dir)
    if ignored_benchmarks:
        print(f"Skipping benchmarks from {bench_ignore_path}:", ", ".join(sorted(ignored_benchmarks)))
        tasks = [task for task in tasks if task not in ignored_benchmarks]

    if not tasks:
        print("No tasks remain after applying .benchignore. Nothing to do.")
        return

    ignored_agents, ignore_path = load_agent_ignore(args.results_dir)
    if ignored_agents:
        print(f"Ignoring agents from {ignore_path}:", ", ".join(sorted(ignored_agents)))

    summary_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    response_presence: Dict[str, Dict[str, bool]] = defaultdict(dict)

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Generating statistics for {task.upper()}")
        print(f"{'='*60}")
        
        # Load results
        agent_results = load_evaluation_results(args.results_dir, task)
        if ignored_agents:
            agent_results = {agent: res for agent, res in agent_results.items() if agent not in ignored_agents}

        response_agents = discover_response_agents(args.results_dir, task)
        for agent_name in response_agents:
            if agent_name in ignored_agents:
                continue
            response_presence[agent_name][task] = True

        for agent_name in agent_results.keys():
            if agent_name in ignored_agents:
                continue
            response_presence[agent_name][task] = True
        
        if not agent_results:
            print(f"No evaluation results found for {task}")
            continue
        
        # Calculate statistics
        stats = calculate_task_statistics(agent_results, task)
        
        # Update summary with llm scores
        for agent_name, agent_stats in stats.items():
            if agent_name in ignored_agents:
                continue
            llm_score = agent_stats.get('llm_score_avg')
            if llm_score is None:
                continue
            summary_scores[agent_name][task] = llm_score

        # Generate tables
        overall_table = generate_overall_table(stats, task)
        category_table = generate_category_table(stats, task)
        
        # Display tables
        tables = []
        if overall_table:
            print("\n" + str(overall_table))
            tables.append(overall_table)
        
        if category_table:
            print("\n" + str(category_table))
            tables.append(category_table)
        
        # Save to file if requested
        if tables and args.save_txt:
            save_tables_to_file(tables, task, args.results_dir)
        
        print(f"\nFound {len(agent_results)} agents with results:")
        for agent, results in agent_results.items():
            print(f"  - {agent}: {len(results)} questions")

    summary_table = generate_summary_table(summary_scores, tasks)
    if summary_table:
        print(f"\n{'='*60}")
        print("Cross-task LLM Score Summary")
        print(f"{'='*60}")
        print(summary_table)
        if args.save_txt:
            save_summary_table(summary_table, args.results_dir)
    else:
        print("\nNo LLM score data available to build the summary table.")

    response_table = generate_response_presence_table(response_presence, tasks)
    if response_table:
        print(f"\n{'='*60}")
        print("Response File Availability")
        print(f"{'='*60}")
        print(response_table)
        if args.save_txt:
            save_response_table(response_table, args.results_dir)
    else:
        print("\nNo response files detected across tasks.")

    # Generate session scaling plot for synth-ss benchmarks
    plot_session_scaling(summary_scores, tasks, args.results_dir)

if __name__ == "__main__":
    main()