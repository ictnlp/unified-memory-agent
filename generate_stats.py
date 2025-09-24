#!/usr/bin/env python3
"""
Generate evaluation statistics tables for each task
"""

import os
import json
import glob
import argparse
from collections import defaultdict
from prettytable import PrettyTable
from datetime import datetime

# Import dataset loaders
import sys
sys.path.append('.')
from data.EvalDataset import load_locomo, load_longmemeval

def parse_args():
    parser = argparse.ArgumentParser(description='Generate evaluation statistics')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory containing task subdirectories')
    parser.add_argument('--task', type=str, choices=['locomo', 'longmemeval', 'all'], default='all',
                        help='Task to generate statistics for')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save statistics to txt file (default: False)')
    return parser.parse_args()

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
        # Extract agent name from filename: evaluated_{agent}_{timestamp}.jsonl
        filename = os.path.basename(file_path)
        try:
            agent_name = filename.split('.')[0].split('_')[1]
        except IndexError:
            agent_name = "unknown"
        
        # Load results
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                agent_results[agent_name].append(result)
    
    return agent_results

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
            
            # Collect all numeric metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
                    # Also add to category stats
                    category_stats[category][metric_name].append(value)
        
        # Calculate averages for overall metrics
        for metric_name, values in all_metrics.items():
            agent_stats[f'{metric_name}_avg'] = sum(values) / len(values) if values else 0
            agent_stats[f'{metric_name}_total'] = sum(values)
        
        # Calculate category-specific averages
        agent_stats['categories'] = {}
        for category, category_metrics in category_stats.items():
            agent_stats['categories'][category] = {}
            for metric_name, values in category_metrics.items():
                agent_stats['categories'][category][f'{metric_name}_avg'] = sum(values) / len(values) if values else 0
                agent_stats['categories'][category][f'{metric_name}_count'] = len(values)
        
        stats[agent_name] = agent_stats
    
    return stats

def build_qid_category_mapping(task):
    """Build mapping from qid to category from original dataset"""
    qid_to_category = {}
    
    if task == 'locomo':
        eval_set = load_locomo()
    elif task == 'longmemeval':
        eval_set = load_longmemeval()
    else:
        return qid_to_category
    
    for sample in eval_set:
        for question in sample.questions:
            if question.qid and question.category is not None:
                qid_to_category[question.qid] = question.category
    
    return qid_to_category

def generate_overall_table(stats, task):
    """Generate overall performance table"""
    if not stats:
        return None
        
    table = PrettyTable()
    table.title = f"{task.upper()} - Overall Performance"
    
    # Determine columns based on available metrics
    agents = list(stats.keys())
    if not agents:
        return None
    
    sample_stats = stats[agents[0]]
    metric_columns = [col for col in sample_stats.keys() 
                     if col.endswith('_avg') and 'categories' not in col]
    
    columns = ['Agent', 'Total Questions'] + [col.replace('_avg', '').upper() for col in metric_columns]
    table.field_names = columns
    
    for agent_name, agent_stats in stats.items():
        row = [agent_name, agent_stats['total_questions']]
        for metric_col in metric_columns:
            value = agent_stats.get(metric_col, 0)
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
    
    # Determine metric types
    sample_agent = list(stats.keys())[0]
    sample_category = list(all_categories)[0]
    sample_category_stats = stats[sample_agent].get('categories', {}).get(sample_category, {})
    metric_types = [col.replace('_avg', '') for col in sample_category_stats.keys() if col.endswith('_avg')]
    
    if not metric_types:
        return None
    
    # Create table with dynamic columns
    columns = ['Agent', 'Category', 'Count'] + [m.upper() for m in metric_types]
    table.field_names = columns
    
    for agent_name, agent_stats in stats.items():
        categories = agent_stats.get('categories', {})
        for category in sorted(all_categories):
            if category in categories:
                cat_stats = categories[category]
                row = [agent_name, str(category), cat_stats.get(f'{metric_types[0]}_count', 0)]
                for metric in metric_types:
                    value = cat_stats.get(f'{metric}_avg', 0)
                    row.append(f"{value:.4f}")
                table.add_row(row)
    
    return table

def save_tables_to_file(tables, task, results_dir):
    """Save tables to file"""
    output_file = os.path.join(results_dir, task, f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Statistics for {task.upper()}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for table in tables:
            if table:
                f.write(str(table) + "\n\n")
    
    print(f"Statistics saved to: {output_file}")

def main():
    args = parse_args()
    
    tasks = ['locomo', 'longmemeval'] if args.task == 'all' else [args.task]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Generating statistics for {task.upper()}")
        print(f"{'='*60}")
        
        # Load results
        agent_results = load_evaluation_results(args.results_dir, task)
        
        if not agent_results:
            print(f"No evaluation results found for {task}")
            continue
        
        # Calculate statistics
        stats = calculate_task_statistics(agent_results, task)
        
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

if __name__ == "__main__":
    main()