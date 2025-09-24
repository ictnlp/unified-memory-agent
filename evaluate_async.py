from openai import OpenAI, RateLimitError
import tqdm
import uuid
import json
import time
import os
import argparse
from datetime import datetime
import backoff
from abc import ABC, abstractmethod
from utils import f1_score, f1_multi_answer
import regex
import string
import unicodedata
from collections import Counter
from bert_score import score
from rouge import Rouge
from nltk.stem import PorterStemmer
import asyncio
from tqdm.asyncio import tqdm as atqdm

# Use uvloop for better async performance if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from config import DATASET_LOADERS, AGENT_CLASS, MODEL_NAME, JUDGE_MODEL_NAME, API_CONFIG, API_CONFIG_LOCAL

class BaseEvaluator(ABC):
    """Base class for task-specific evaluators"""
    
    def __init__(self):
        self.qid_category_map = {}
        self.ps = PorterStemmer()
        self.rouge = Rouge()
    
    def normalize_answer(self, s):
        """Normalize answer text for evaluation"""
        s = s.replace(',', "")
        
        def remove_articles(text):
            return regex.sub(r'\\b(a|an|the|and)\\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, prediction, ground_truth):
        """Calculate exact match score"""
        prediction = self.normalize_answer(prediction)
        ground_truth = self.normalize_answer(ground_truth)
        return 1.0 if set(prediction.split()) == set(ground_truth.split()) else 0.0
    
    def bert_score(self, prediction, ground_truth):
        """Calculate BERT score using default device (async mode supports GPU)"""
        prediction = self.normalize_answer(prediction)
        ground_truth = self.normalize_answer(ground_truth)
        # Use default device in async mode
        P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, 
                         rescale_with_baseline=True)
        return max(0, F1[0].item())
    
    def rouge_score(self, prediction, ground_truth):
        """Calculate ROUGE-L score"""
        prediction = ' '.join([self.ps.stem(w) for w in self.normalize_answer(prediction).split()])
        ground_truth = ' '.join([self.ps.stem(w) for w in self.normalize_answer(ground_truth).split()])
        try:
            scores = self.rouge.get_scores(prediction, ground_truth, avg=True)
            return scores["rouge-l"]["f"]
        except ValueError:  # "Hypothesis is empty."
            return 0.0
    
    def f1_score_custom(self, prediction, ground_truth):
        """Calculate F1 score with stemming"""
        prediction_tokens = [self.ps.stem(w) for w in self.normalize_answer(prediction).split()]
        ground_truth_tokens = [self.ps.stem(w) for w in self.normalize_answer(ground_truth).split()]
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
        
    def set_category_mapping(self, eval_set):
        """Build mapping from qid to category from original dataset"""
        for sample in eval_set:
            for question in sample.questions:
                if question.qid and question.category is not None:
                    self.qid_category_map[question.qid] = question.category
    
    @abstractmethod
    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        """Evaluate a single QA pair and return metrics dictionary"""
        pass

class LongmemEvalEvaluator(BaseEvaluator):
    """Async Evaluator for LongmemEval dataset"""
    
    # Scoring templates
    TEMPLATES = {
        'default': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \\n\\nQuestion: {}\\n\\nCorrect Answer: {}\\n\\nModel Response: {}\\n\\nIs the model response correct? Answer yes or no only.",
        'temporal-reasoning': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \\n\\nQuestion: {}\\n\\nCorrect Answer: {}\\n\\nModel Response: {}\\n\\nIs the model response correct? Answer yes or no only.",
        'knowledge-update': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\\n\\nQuestion: {}\\n\\nCorrect Answer: {}\\n\\nModel Response: {}\\n\\nIs the model response correct? Answer yes or no only.",
        'single-session-preference': "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\\n\\nQuestion: {}\\n\\nRubric: {}\\n\\nModel Response: {}\\n\\nIs the model response correct? Answer yes or no only.",
        'abstention': "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\\n\\nQuestion: {}\\n\\nExplanation: {}\\n\\nModel Response: {}\\n\\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    }
    
    def __init__(self, scoring_client: OpenAI):
        super().__init__()
        self.scoring_client = scoring_client
    
    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=16, max_time=300, jitter=backoff.full_jitter)
    async def _llm_scoring_response(self, messages, model_name=JUDGE_MODEL_NAME):
        # Use asyncio.to_thread to run blocking API call in thread
        response = await asyncio.to_thread(
            lambda: self.scoring_client.chat.completions.create(
                model=model_name, messages=messages, temperature=0, max_tokens=10
            ).choices[0].message.content
        )
        return response
    
    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        try:
            question_type = self.qid_category_map.get(qid, "multi-session")
            
            # Additional metrics
            f1_custom = self.f1_score_custom(response, expected_answer)
            exact_match = self.exact_match_score(response, expected_answer)
            bert_score_val = self.bert_score(response, expected_answer)
            rouge_score_val = self.rouge_score(response, expected_answer)
            
            # Check if abstention task
            qid_parts = qid.split('_')
            abstention = len(qid_parts) >= 3 and '_'.join(qid_parts[1:-1]).endswith("_abs")
            # Get appropriate template
            if abstention:
                template = self.TEMPLATES['abstention']
            else:
                template = self.TEMPLATES.get(question_type, self.TEMPLATES['default'])
            
            try:
                prompt = template.format(query, expected_answer, response)
                messages = [{"role": "user", "content": prompt}]
                scoring_result = await self._llm_scoring_response(messages)
                llm_score = 1 if 'yes' in scoring_result.lower() and 'no' not in scoring_result.lower() else 0
            except Exception as e:
                print(f"Error in LLM scoring for qid {qid}: {e}")
                llm_score = -1  # -1 indicates LLM scoring error
            
            return {
                'f1_score': f1_custom,
                'exact_match': exact_match,
                'bert_score': bert_score_val,
                'rouge_score': rouge_score_val,
                'llm_score': llm_score
            }
            
        except Exception as e:
            print(f"Error evaluating LongmemEval QA for qid {qid}: {e}")
            # Use -1 to indicate evaluation errors, distinguish from 0 (wrong answer)
            return {'f1_score': -1, 'exact_match': -1, 'bert_score': -1, 'rouge_score': -1, 'llm_score': -1}

def get_evaluator(task: str, scoring_client: OpenAI|None = None) -> BaseEvaluator:
    """Factory function to get appropriate evaluator for task"""
    if task == 'longmemeval':
        if scoring_client is None:
            raise ValueError("scoring_client is required for longmemeval task")
        return LongmemEvalEvaluator(scoring_client)
    else:
        raise NotImplementedError(f"Evaluator for task '{task}' not implemented")

def create_client():
    """Create OpenAI client with standard configuration"""
    return OpenAI(**API_CONFIG_LOCAL)

def create_judge_client():
    """Create OpenAI client for judging"""
    return OpenAI(**API_CONFIG)

def load_dataset(task: str, force_rebuild: bool = False):
    """Load dataset for given task"""
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    return DATASET_LOADERS[task](force_rebuild, oracle=True)

async def process_single_sample(sample, agent, output_file, task, semaphore):
    """Process a single sample asynchronously"""
    async with semaphore:
        print(f"Processing sample: {sample.task_id}")
        
        # Group questions by position
        position_groups = {}
        for question in sample.questions:
            position_groups.setdefault(question.position, []).append(question)
        
        chunk_idx = 0
        results = []
        
        # Process each position in order
        for position in sorted(position_groups.keys()):
            # Add memory chunks up to current position
            while chunk_idx <= position and chunk_idx < len(sample.chunks):
                agent.add_memory(sample.chunks[chunk_idx])
                chunk_idx += 1
            
            # Process questions at this position
            questions_at_position = position_groups[position]
            queries = [q.query for q in questions_at_position]
            
            # Time the QA batch processing
            batch_start_time = time.time()
            responses = agent.QA_batch(queries)
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            # Store results
            for i, (q, response) in enumerate(zip(questions_at_position, responses)):
                per_question_time = batch_duration / len(questions_at_position)
                
                result = {
                    'qid': q.qid,
                    'query': q.query,
                    'expected_answer': q.answer,
                    'response': response,
                    'generation_time': per_question_time
                }
                results.append(result)
        
        # Write all results to file at once (with file lock for thread safety)
        async with asyncio.Lock():
            with open(output_file, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
        
        return len(results)

async def generate_responses_async(task, agent_class, agent_config, agent_id, output_dir, concurrency=32):
    """Generate responses using specified agent with async concurrency"""
    print(f"Starting async generation for task: {task}, agent_id: {agent_id}")
    start_time = time.time()
    
    # Load dataset
    eval_set = load_dataset(task)
    
    # Create output file
    final_output_file = f"{output_dir}/responses_{agent_id}.jsonl"
    
    # Check for existing partial results
    completed_qids = set()
    if os.path.exists(final_output_file):
        print(f"Found existing output file: {final_output_file}")
        with open(final_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    completed_qids.add(result['qid'])
                except:
                    continue
        print(f"Found {len(completed_qids)} already completed questions")
    
    # Filter out completed samples
    remaining_samples = []
    for sample in eval_set:
        remaining_questions = [q for q in sample.questions if q.qid not in completed_qids]
        if remaining_questions:
            remaining_sample = type(sample)(
                task_id=sample.task_id,
                chunks=sample.chunks,
                questions=remaining_questions
            )
            remaining_samples.append(remaining_sample)
    
    if not remaining_samples:
        print("All questions already completed!")
        return final_output_file
    
    print(f"Processing {len(remaining_samples)} remaining samples with concurrency {concurrency}")
    
    # Create semaphore to limit concurrent samples
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create agent instances for concurrent processing
    client = create_client()
    
    # Create tasks for all samples
    tasks = []
    for sample in remaining_samples:
        # Create a new agent instance for each sample to avoid state conflicts
        if agent_class.__name__ == 'FileMemoryAgent':
            agent = agent_class(client=client, task_type=task, task_id=sample.task_id)
        else:
            if 'wo_q' in agent_config:
                agent = agent_class(wo_q=agent_config['wo_q'], client=client)
            else:
                agent = agent_class(model_name="qwen3", client=client)
        
        task_coroutine = process_single_sample(sample, agent, final_output_file, task, semaphore)
        tasks.append(task_coroutine)
    
    # Process all tasks concurrently with progress bar
    results = await atqdm.gather(*tasks, desc="Processing samples")
    
    total_processed = sum(results)
    total_time = time.time() - start_time
    
    print(f"Generation complete! Total time: {total_time:.2f}s")
    print(f"Processed {total_processed} questions across {len(remaining_samples)} samples")
    print(f"Average time per question: {total_time/total_processed:.2f}s" if total_processed > 0 else "No new questions processed")
    print(f"Results saved to: {final_output_file}")
    
    return final_output_file

async def evaluate_single_response(item, evaluator, results_file, semaphore):
    """Evaluate a single response asynchronously"""
    async with semaphore:
        eval_start_time = time.time()
        
        metrics_info = await evaluator.evaluate_qa(
            qid=item['qid'], query=item['query'],
            expected_answer=item['expected_answer'], response=item['response']
        )
        
        eval_end_time = time.time()
        evaluation_time = eval_end_time - eval_start_time
        
        metrics = {k: v for k, v in metrics_info.items() if isinstance(v, (int, float))}
        
        result = {
            'qid': item['qid'],
            'query': item['query'],
            'expected_answer': item['expected_answer'],
            'response': item['response'],
            'metric': metrics,
            'evaluation_time': evaluation_time
        }
        
        # Add generation_time if it exists
        if 'generation_time' in item:
            result['generation_time'] = item['generation_time']
        
        # Write result to file with async lock
        async with asyncio.Lock():
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
        
        return result

async def evaluate_responses_async(input_file, task, output_dir, agent_id="unknown", concurrency=64):
    """Evaluate responses from input file with async concurrency"""
    print(f"Starting async evaluation for file: {input_file}")
    start_time = time.time()
    
    # Load responses
    with open(input_file, 'r', encoding='utf-8') as f:
        responses = [json.loads(line) for line in f]
    
    # Create output file
    results_file = f"{output_dir}/evaluated_{agent_id}.jsonl"
    
    # Check for existing partial results
    completed_qids = set()
    if os.path.exists(results_file):
        print(f"Found existing evaluation file: {results_file}")
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    completed_qids.add(result['qid'])
                except:
                    continue
        print(f"Found {len(completed_qids)} already evaluated questions")
    
    # Filter out completed responses
    remaining_responses = [r for r in responses if r['qid'] not in completed_qids]
    
    if not remaining_responses:
        print("All responses already evaluated!")
        return results_file, {}
    
    print(f"Evaluating {len(remaining_responses)} remaining responses with concurrency {concurrency}")
    
    # Create evaluator and semaphore
    try:
        evaluator = get_evaluator(task, create_judge_client())
        eval_set = load_dataset(task)
        evaluator.set_category_mapping(eval_set)
    except NotImplementedError as e:
        print(f"Warning: {e}. Using default metrics.")
        return results_file, {}
    
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create tasks for all evaluations
    tasks = [
        evaluate_single_response(item, evaluator, results_file, semaphore)
        for item in remaining_responses
    ]
    
    # Process all tasks concurrently with progress bar
    results = await atqdm.gather(*tasks, desc="Evaluating responses")
    
    # Calculate metrics
    total_metrics = {}
    total_evaluation_time = sum(r['evaluation_time'] for r in results)
    
    # Aggregate metrics
    for result in results:
        metrics = result.get('metric', {})
        for metric_name, score in metrics.items():
            if score != -1:
                total_metrics[metric_name] = total_metrics.get(metric_name, 0) + score
                if f"{metric_name}_count" not in total_metrics:
                    total_metrics[f"{metric_name}_count"] = 0
                total_metrics[f"{metric_name}_count"] += 1
    
    # Calculate and display averages
    avg_metrics = {}
    total_time = time.time() - start_time
    avg_evaluation_time = total_evaluation_time / len(results) if results else 0
    
    print(f"Evaluation complete! Metrics:")
    
    for metric_name in ['f1_score', 'exact_match', 'bert_score', 'rouge_score', 'llm_score']:
        if metric_name in total_metrics:
            valid_count = total_metrics.get(f"{metric_name}_count", 0)
            if valid_count > 0:
                avg_score = total_metrics[metric_name] / valid_count
                avg_metrics[metric_name] = avg_score
                print(f"  {metric_name}: {avg_score:.4f} ({total_metrics[metric_name]}/{valid_count} valid)")
    
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Average evaluation time per question: {avg_evaluation_time:.2f}s")
    print(f"Results saved to: {results_file}")
    
    return results_file, avg_metrics

def get_args():
    parser = argparse.ArgumentParser(description='Memory Agent Evaluation - Async Version')
    parser.add_argument('--task', type=str, choices=['locomo', 'longmemeval'], default='longmemeval')
    parser.add_argument('--agent', type=str, choices=['concat', 'memagent', 'filememory', 'emergence'], default='emergence')
    parser.add_argument('--wo_q', action='store_true', help='Use wo_q mode for MemAgent')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file for evaluation only (JSONL format)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/{task})')
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Number of concurrent tasks (default: 32)')
    parser.add_argument('--generate_only', action='store_true', help='Generate Only mode')
    return parser.parse_args()

async def main():
    args = get_args()
    
    # Setup output directory
    output_dir = args.output_dir or f"results/{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.input_file is None:
        # Generation + Evaluation
        print("Mode: Async Generation + Evaluation")
        
        # Create agent configuration
        if args.agent in AGENT_CLASS:
            agent_class = AGENT_CLASS[args.agent]
            if args.agent == 'memagent' and args.wo_q:
                agent_config = {'wo_q': True}
                agent_id = 'memagent_woq'
            else:
                agent_config = {}
                agent_id = args.agent
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        
        responses_file = await generate_responses_async(
            args.task, agent_class, agent_config, agent_id, output_dir, args.concurrency
        )
        
        if not args.generate_only:
            await evaluate_responses_async(
                responses_file, args.task, output_dir, agent_id, args.concurrency
            )
    else:
        # Evaluation only
        print("Mode: Async Evaluation only")
        agent_id = "unknown"
        if "responses_" in args.input_file:
            try:
                filename_part = args.input_file.split("responses_")[1]
                if "_woq_" in filename_part:
                    agent_id = filename_part.split("_woq_")[0] + "_woq"
                else:
                    agent_id = filename_part.split("_")[0]
            except:
                pass
        
        await evaluate_responses_async(
            args.input_file, args.task, output_dir, agent_id, args.concurrency
        )

if __name__ == "__main__":
    asyncio.run(main())