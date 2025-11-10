# import debugpy
# debugpy.listen(("0.0.0.0", 5678))  # 监听所有 IP，端口可改 
# print("  Waiting for debugger attach on port 5678...") 
# debugpy.wait_for_client()  # 等待调试器连接
from openai import OpenAI, AsyncOpenAI, RateLimitError
import tqdm
import uuid
import json
import time
import os
from pathlib import Path
import argparse
from datetime import datetime
import backoff
from abc import ABC, abstractmethod
import regex
import string
import unicodedata
from collections import Counter
from rouge_score import rouge_scorer
import asyncio
from tqdm.asyncio import tqdm as atqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

# Use uvloop for better async performance if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from config import DATASET_LOADERS, AGENT_CLASS, MODEL_NAME, JUDGE_MODEL_NAME, API_CONFIG, API_CONFIG_LOCAL

API_CONFIG = API_CONFIG_LOCAL
# JUDGE_MODEL_NAME = MODEL_NAME
# JUDGE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

class BaseEvaluator(ABC):
    """Base class for task-specific evaluators"""
    LLM_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    # Class-level singleton for bert_score model components
    _bert_scorer = None
    _bert_device = None
    _bert_lock = None
    _bert_available = True
    
    def __init__(self):
        self.qid_category_map = {}
        self.scoring_client: Optional[AsyncOpenAI] = None
        # Pre-initialize ROUGE scorer; use stemmer-enabled tokenizer for consistency with Porter stemming
        self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=16, max_time=300, jitter=backoff.full_jitter)
    async def _llm_scoring_response(self, messages, model_name=JUDGE_MODEL_NAME):
        response = await self.scoring_client.chat.completions.create(
            model=model_name, messages=messages, temperature=0, max_tokens=2048
        )
        content = response.choices[0].message.content
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content

    @classmethod
    def warmup_bert_model(cls):
        """Initialize BERT scorer in main thread with proper locking"""
        if cls._bert_scorer is None:
            import threading
            if cls._bert_lock is None:
                cls._bert_lock = threading.Lock()
            
            with cls._bert_lock:
                if cls._bert_scorer is None:
                    try:
                        print("Loading BERT model...")
                        import torch
                        from bert_score import BERTScorer

                        cls._bert_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        # Create a single BERTScorer instance that will be reused
                        cls._bert_scorer = BERTScorer(
                            lang='en',
                            rescale_with_baseline=True,
                            device=cls._bert_device
                        )
                        print(f"BERT model loaded on {cls._bert_device}")
                    except Exception as exc:  # pragma: no cover - environment specific failures
                        cls._bert_available = False
                        print(
                            "Warning: failed to initialize BERTScorer, disabling bert_score metric. "
                            f"Error: {exc}"
                        )
                        return
            cls._bert_available = cls._bert_scorer is not None
    
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
    
    def compute_token_metrics(self, prediction, ground_truth):
        """
        Compute all token-based metrics in one pass.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
        
        Returns dict with:
        - f1_score: Token-level F1 score
        - precision: Token-level precision
        - recall: Token-level recall
        - exact_match: Set-based exact match
        - sub_em: Substring exact match
        """
        # Standard token-based metrics
        pred_norm = self.normalize_answer(prediction)
        gold_norm = self.normalize_answer(ground_truth)
        
        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()
        
        metrics = {}
        
        # Basic token-level metrics (no stemming)
        if not pred_tokens or not gold_tokens:
            metrics['f1_score'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
        else:
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
            else:
                metrics['precision'] = 1.0 * num_same / len(pred_tokens)
                metrics['recall'] = 1.0 * num_same / len(gold_tokens)
                metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        # Exact match (set-based)
        metrics['exact_match'] = 1.0 if set(pred_tokens) == set(gold_tokens) else 0.0
        
        # Sub exact match (substring match)
        metrics['sub_em'] = 1.0 if (pred_norm in gold_norm or gold_norm in pred_norm) else 0.0
        
        return metrics
    
    def _bert_score_sync(self, prediction, ground_truth):
        """Synchronous BERT score calculation using singleton scorer"""
        prediction = self.normalize_answer(prediction)
        ground_truth = self.normalize_answer(ground_truth)
        if not BaseEvaluator._bert_available or BaseEvaluator._bert_scorer is None:
            return 0.0

        # Use the pre-initialized singleton scorer with thread safety
        with BaseEvaluator._bert_lock:
            P, R, F1 = BaseEvaluator._bert_scorer.score([prediction], [ground_truth])
        return max(0, F1[0].item())
    
    async def bert_score(self, prediction, ground_truth):
        """Calculate BERT score using thread pool to avoid blocking"""
        return await asyncio.to_thread(self._bert_score_sync, prediction, ground_truth)
    
    def rouge_score(self, prediction, ground_truth):
        """Calculate ROUGE-L F1 score using an iterative LCS implementation."""
        prediction_norm = self.normalize_answer(str(prediction))
        ground_truth_norm = self.normalize_answer(str(ground_truth))

        if not prediction_norm or not ground_truth_norm:
            return 0.0
        scores = self._rouge_scorer.score(ground_truth_norm, prediction_norm)
        return scores["rougeL"].fmeasure
        
    def set_category_mapping(self, eval_set):
        """Build mapping from qid to category from original dataset"""
        for sample in eval_set:
            for question in sample.questions:
                if question.qid and question.category is not None:
                    self.qid_category_map[question.qid] = question.category

    async def _safe_metric(self, metric_name: str, func: Callable[[], Any], qid: Optional[str] = None, default_keys: Optional[List[str]] = None):
        """
        Execute a metric function (sync or async) and handle errors gracefully.
        
        Args:
            metric_name: Name of the metric being computed
            func: Function to execute (can be sync or async)
            qid: Question ID for logging
            default_keys: If provided and func returns dict, use these keys for -1 fallback on error
        
        Returns:
            - Result from func() if successful
            - Dict with default_keys set to -1 if func fails and default_keys provided
            - -1 if func fails and no default_keys provided
        """
        try:
            result = func()
            # Handle async functions
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, dict) and default_keys:
                return {key: -1 for key in default_keys}
            return result
        except Exception as exc:
            suffix = f" for qid {qid}" if qid else ""
            print(f"Error computing {metric_name}{suffix}: {exc}")
            if default_keys:
                return {key: -1 for key in default_keys}
            return -1
    
    async def _token_metrics(self, qid: str, response_text: str, expected_text: str,
                             *, default_keys: Optional[List[str]] = None) -> Dict[str, float]:
        """Wrapper around compute_token_metrics with unified error handling."""
        return await self._safe_metric(
            'token_metrics',
            lambda: self.compute_token_metrics(response_text, expected_text),
            qid=qid,
            default_keys=default_keys
        )

    async def _bert_metric(self, qid: str, response_text: str, expected_text: str) -> float:
        """Compute BERTScore F1 with error handling."""
        return await self._safe_metric(
            'bert_score',
            lambda: self.bert_score(response_text, expected_text),
            qid=qid
        )

    async def _rouge_metric(self, qid: str, response_text: str, expected_text: str) -> float:
        """Compute ROUGE-L F1 with error handling."""
        return await self._safe_metric(
            'rouge_score',
            lambda: self.rouge_score(response_text, expected_text),
            qid=qid
        )

    async def _llm_metric(self, qid: str, query: str, expected_text: str, response_text: str,
                           *, template: Optional[str] = None) -> Optional[float]:
        """Run LLM-based grading if scoring client is available."""
        if not self.scoring_client:
            return None
        prompt_template = template or self.LLM_TEMPLATE
        try:
            prompt = prompt_template.format(query, expected_text, response_text)
            messages = [{"role": "user", "content": prompt}]
            scoring_result = await self._llm_scoring_response(messages)
            normalized = scoring_result.lower()
            return 1.0 if ('yes' in normalized and 'no' not in normalized) else 0.0
        except Exception as exc:
            print(f"Error in LLM scoring for qid {qid}: {exc}")
            return -1.0

    @abstractmethod
    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        """Evaluate a single QA pair and return metrics dictionary"""
        pass

class LocomoEvaluator(BaseEvaluator):
    """Async Evaluator for Locomo dataset"""
    
    def __init__(self, scoring_client: OpenAI):
        super().__init__()
        self.scoring_client = scoring_client
    
    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        metrics: Dict[str, float] = {}
        try:
            response_text = str(response)
            expected_text = str(expected_answer)
            
            token_metrics = await self._token_metrics(
                qid,
                response_text,
                expected_text,
                default_keys=['f1_score', 'precision', 'recall', 'exact_match', 'sub_em']
            )
            metrics.update(token_metrics)

            metrics['bert_score'] = await self._bert_metric(qid, response_text, expected_text)
            metrics['rouge_score'] = await self._rouge_metric(qid, response_text, expected_text)

            llm_score = await self._llm_metric(qid, query, expected_text, response_text)
            if llm_score is not None:
                metrics['llm_score'] = llm_score

            return metrics

        except Exception as e:
            print(f"Error evaluating Locomo QA for qid {qid}: {e}")
            failure_keys = ['f1_score', 'precision', 'recall', 'exact_match', 'sub_em', 'bert_score', 'rouge_score']
            fallback = {key: metrics.get(key, -1) for key in failure_keys}
            if self.scoring_client:
                fallback['llm_score'] = metrics.get('llm_score', -1)
            return fallback

class HotpotQAEvaluator(BaseEvaluator):
    """Async Evaluator for HotpotQA dataset"""

    def __init__(self, scoring_client: OpenAI = None):
        super().__init__()
        self.scoring_client = scoring_client

    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        metrics: Dict[str, float] = {}
        try:
            response_text = str(response)
            expected_text = str(expected_answer)

            token_metrics = await self._token_metrics(
                qid,
                response_text,
                expected_text,
                default_keys=['f1_score', 'precision', 'recall', 'exact_match', 'sub_em']
            )
            metrics.update(token_metrics)

            metrics['bert_score'] = await self._bert_metric(qid, response_text, expected_text)
            metrics['rouge_score'] = await self._rouge_metric(qid, response_text, expected_text)

            llm_score = await self._llm_metric(qid, query, expected_text, response_text)
            if llm_score is not None:
                metrics['llm_score'] = llm_score

            return metrics

        except Exception as e:
            print(f"Error evaluating HotpotQA for qid {qid}: {e}")
            failure_keys = ['f1_score', 'precision', 'recall', 'exact_match', 'sub_em', 'bert_score', 'rouge_score']
            fallback = {key: metrics.get(key, -1) for key in failure_keys}
            if self.scoring_client:
                fallback['llm_score'] = metrics.get('llm_score', -1)
            return fallback

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

    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        metrics: Dict[str, float] = {}
        try:
            question_type = self.qid_category_map.get(qid, "multi-session")
            response_text = str(response)
            expected_text = str(expected_answer)

            token_metrics = await self._token_metrics(
                qid,
                response_text,
                expected_text,
                default_keys=['f1_score', 'precision', 'recall', 'exact_match', 'sub_em']
            )
            metrics.update(token_metrics)

            metrics['bert_score'] = await self._bert_metric(qid, response_text, expected_text)
            metrics['rouge_score'] = await self._rouge_metric(qid, response_text, expected_text)

            qid_parts = qid.split('_')
            abstention = len(qid_parts) >= 3 and '_'.join(qid_parts[1:-1]).endswith("_abs")
            if abstention:
                template = self.TEMPLATES['abstention']
            else:
                template = self.TEMPLATES.get(question_type, self.TEMPLATES['default'])

            llm_score = await self._llm_metric(qid, query, expected_text, response_text, template=template)
            if llm_score is not None:
                metrics['llm_score'] = llm_score
            return metrics

        except Exception as e:
            print(f"Error evaluating LongmemEval QA for qid {qid}: {e}")
            failure_keys = ['f1_score', 'precision', 'recall', 'exact_match', 'sub_em', 'bert_score', 'rouge_score', 'llm_score']
            return {key: metrics.get(key, -1) for key in failure_keys}


class MSCEvaluator(BaseEvaluator):
    """Evaluator for MSC-Self-Instruct dataset"""
    def __init__(self, scoring_client: OpenAI):
        super().__init__()
        self.scoring_client = scoring_client
    
    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        """
        Evaluate a single QA pair for MSC-Self-Instruct
        Uses token-based metrics + BERT score for memory recall tasks
        """
        metrics: Dict[str, float] = {}
        try:
            response_text = str(response)
            expected_text = str(expected_answer)
            
            # Token-based metrics
            token_metrics = await self._token_metrics(
                qid,
                response_text,
                expected_text,
                default_keys=['f1_score', 'precision', 'recall', 'exact_match', 'sub_em']
            )
            metrics.update(token_metrics)
            
            metrics['bert_score'] = await self._bert_metric(qid, response_text, expected_text)
            metrics['rouge_score'] = await self._rouge_metric(qid, response_text, expected_text)

            llm_score = await self._llm_metric(qid, query, expected_text, response_text)
            if llm_score is not None:
                metrics['llm_score'] = llm_score

            return metrics
            
        except Exception as e:
            print(f"Error evaluating MSC QA for qid {qid}: {e}")
            failure_keys = ['f1_score', 'precision', 'recall', 'exact_match', 'sub_em', 'bert_score', 'rouge_score', 'llm_score']
            return {key: metrics.get(key, -1) for key in failure_keys}

def get_evaluator(task: str, scoring_client: OpenAI|None = None) -> BaseEvaluator:
    """Factory function to get appropriate evaluator for task"""
    if task == 'longmemeval':
        if scoring_client is None:
            raise ValueError("scoring_client is required for longmemeval task")
        return LongmemEvalEvaluator(scoring_client)
    elif task == 'locomo':
        return LocomoEvaluator(scoring_client)
    elif task == 'hotpotqa':
        return HotpotQAEvaluator(scoring_client)
    elif task == 'msc':
        return MSCEvaluator(scoring_client)
    else:
        raise NotImplementedError(f"Evaluator for task '{task}' not implemented")

def create_client():
    """Create AsyncOpenAI client for async operations"""
    return AsyncOpenAI(**API_CONFIG_LOCAL)

def create_judge_client():
    """Create AsyncOpenAI client for judging"""
    return AsyncOpenAI(**API_CONFIG)

def load_dataset(task: str, force_rebuild: bool = False):
    """Load dataset for given task"""
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    return DATASET_LOADERS[task](force_rebuild)

async def process_single_sample(sample, agent, output_file, task, semaphore):
    """Process a single sample asynchronously"""
    async with semaphore:
        print(f"Processing sample: {sample.task_id}")

        if hasattr(agent, "reset"):
            agent.reset()
        if hasattr(agent, "prepare_sample"):
            agent.prepare_sample(sample)
        
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
                # Check if agent has async add_memory method
                if hasattr(agent, 'add_memory_async'):
                    await agent.add_memory_async(sample.chunks[chunk_idx])
                else:
                    agent.add_memory(sample.chunks[chunk_idx])
                chunk_idx += 1
            
            # Process questions at this position
            questions_at_position = position_groups[position]
            queries = [q.query for q in questions_at_position]
            
            # Time the QA batch processing
            batch_start_time = time.time()
            responses = await agent.QA_batch_async(queries, batch_size=5)
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

async def generate_responses_async(task, agent_class, agent_config, agent_id, output_dir, concurrency=32, model_name='qwen3-4b'):
    """Generate responses using specified agent with async concurrency"""
    print(f"Starting async generation for task: {task}, agent_id: {agent_id}, model: {model_name}")
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
        elif agent_class.__name__ == 'MemAgent':
            # MemAgent uses wo_q parameter
            wo_q = agent_config.get('wo_q', False)
            agent = agent_class(wo_q=wo_q, client=client, model_name=model_name)
        elif agent_class.__name__ == 'MemAlphaUnifiedAgent':
            agent = agent_class(client=client, **agent_config)
        else:
            # Other agents (ConcatAgent, EmergenceAgent) use model_name
            agent = agent_class(model_name=model_name, client=client)
        
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
            expected_answer=str(item['expected_answer']), response=str(item['response'])
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
    valid_results = []  # Store results with all metrics valid
    if os.path.exists(results_file):
        print(f"Found existing evaluation file: {results_file}")
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    qid = result['qid']
                    metrics = result.get('metric', {})
                    
                    # Check if all metrics are valid (not -1)
                    all_metrics_valid = all(v != -1 for v in metrics.values()) if metrics else False
                    
                    if all_metrics_valid:
                        completed_qids.add(qid)
                        valid_results.append(result)
                    else:
                        print(f"Found incomplete evaluation for qid {qid}, will re-evaluate")
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        # Rewrite the file with only valid results
        if valid_results:
            with open(results_file, 'w', encoding='utf-8') as f:
                for result in valid_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Cleaned results file: kept {len(valid_results)} valid evaluations")
        elif os.path.exists(results_file):
            # If no valid results, remove the file
            os.remove(results_file)
            print("No valid evaluations found, removed results file")
        
        print(f"Found {len(completed_qids)} completed questions with valid metrics")
    
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
        
        # Warmup BERT model in main thread before starting concurrent evaluation
        BaseEvaluator.warmup_bert_model()
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
    
    for metric_name in ['f1_score', 'exact_match', 'bert_score', 'rouge_score', 'llm_score', 'precision', 'recall', 'sub_em']:
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
    parser.add_argument('--task', type=str, choices=['locomo', 'longmemeval', 'hotpotqa', 'msc', 'memalpha', 'trec_coarse', 'trec_fine', 'banking77', 'clinic', 'nlu', 'booksum', 'perltqa', 'pubmed_rct'], default='longmemeval')
    parser.add_argument('--agent', type=str, choices=['concat', 'memagent', 'filememory', 'emergence', 'memalpha'], default='emergence')
    parser.add_argument('--wo_q', action='store_true', help='Use wo_q mode for MemAgent')
    default_agent_cfg_path = (
        Path(__file__).resolve().parents[1]
        / 'Mem-alpha'
        / 'config'
        / 'memalpha-qwen3-4b_agent_0.05-0.1.yaml'
    )
    default_agent_cfg = str(default_agent_cfg_path) if default_agent_cfg_path.exists() else None
    parser.add_argument('--agent_config_path', type=str, default=default_agent_cfg,
                        help='Optional agent configuration file (used by memalpha agent)')
    default_prompts_cfg_path = (
        Path(__file__).resolve().parents[1]
        / 'Mem-alpha'
        / 'config'
        / 'prompts_wrt_datasource.yaml'
    )
    default_prompts_cfg = str(default_prompts_cfg_path) if default_prompts_cfg_path.exists() else None
    parser.add_argument('--prompts_config_path', type=str, default=default_prompts_cfg,
                        help='Optional prompts configuration file for memalpha agent')
    parser.add_argument('--model', type=str, default='qwen3-4b',
                        help='Model name to use (default: qwen3-4b)')
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
    
    # Increase thread pool size for high concurrency
    import concurrent.futures
    loop = asyncio.get_event_loop()
    max_workers = max(64, args.concurrency // 2)  # At least 64 threads, or half of concurrency
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    loop.set_default_executor(executor)
    print(f"Set ThreadPoolExecutor with {max_workers} workers for concurrency {args.concurrency}")
    
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
            elif args.agent == 'memalpha':
                agent_config = {}
                if args.agent_config_path:
                    agent_config['agent_config_path'] = args.agent_config_path
                if args.prompts_config_path:
                    agent_config['prompts_config_path'] = args.prompts_config_path
                agent_id = args.agent
            else:
                agent_config = {}
                agent_id = args.agent
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        
        responses_file = await generate_responses_async(
            args.task, agent_class, agent_config, agent_id, output_dir, args.concurrency, args.model
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
                filename_part = args.input_file.split("responses_")[1].replace(".jsonl", "")
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