import debugpy
debugpy.listen(("0.0.0.0", 5678))  # 监听所有 IP，端口可改 
print("  Waiting for debugger attach on port 5678...") 
debugpy.wait_for_client()  # 等待调试器连接
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
import regex
import string
import unicodedata
from collections import Counter, defaultdict
from rouge_score import rouge_scorer
import asyncio
from tqdm.asyncio import tqdm as atqdm
import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

# Use uvloop for better async performance if available
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from config import DATASET_LOADERS, AGENT_CLASS, API_CONFIG_LOCAL, API_CONFIG_REMOTE

#如果存在环境变量OPENAI_API_BASE
if 'OPENAI_API_BASE' in os.environ:
    API_CONFIG_LOCAL['base_url'] = os.environ['OPENAI_API_BASE']

JUDGE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

class BaseEvaluator:
    """Base class for task-specific evaluators"""
    LLM_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    # Class-level singleton for bert_score model components
    _bert_scorer = None
    _bert_device = None
    _bert_lock = None
    _bert_available = True
    
    def __init__(self, scoring_client: OpenAI = None):
        self.qid_category_map = {}
        self.scoring_client: Optional[AsyncOpenAI] = None
        # Pre-initialize ROUGE scorer; use stemmer-enabled tokenizer for consistency with Porter stemming
        self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.scoring_client = scoring_client

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

    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str, existing_metrics: dict = None) -> dict:
        existing_metrics = existing_metrics or {}
        metrics: Dict[str, float] = {}
        try:
            response_text = str(response)
            expected_text = str(expected_answer)

            # Token metrics - always recompute (cheap)
            token_metrics = await self._token_metrics(
                qid,
                response_text,
                expected_text,
                default_keys=['f1_score', 'precision', 'recall', 'exact_match', 'sub_em']
            )
            metrics.update(token_metrics)

            # BERT score - skip if already computed (expensive)
            if 'bert_score' in existing_metrics and existing_metrics['bert_score'] != -1:
                metrics['bert_score'] = existing_metrics['bert_score']
                print(f"Reusing bert_score for qid {qid}")
            else:
                metrics['bert_score'] = await self._bert_metric(qid, response_text, expected_text)

            # ROUGE score - always recompute (cheap)
            metrics['rouge_score'] = await self._rouge_metric(qid, response_text, expected_text)

            # LLM score - skip if already computed (expensive)
            if 'llm_score' in existing_metrics and existing_metrics['llm_score'] != -1:
                metrics['llm_score'] = existing_metrics['llm_score']
                print(f"Reusing llm_score for qid {qid}")
            else:
                llm_score = await self._llm_metric(qid, query, expected_text, response_text)
                if llm_score is not None:
                    metrics['llm_score'] = llm_score

            return metrics

        except Exception as e:
            print(f"Error evaluating for qid {qid}: {e}")
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

    def _get_template_for_qid(self, qid: str) -> str:
        """Get appropriate template based on qid and question type"""
        question_type = self.qid_category_map.get(qid, "multi-session")
        qid_parts = qid.split('_')
        abstention = len(qid_parts) >= 3 and '_'.join(qid_parts[1:-1]).endswith("_abs")
        if abstention:
            return self.TEMPLATES['abstention']
        else:
            return self.TEMPLATES.get(question_type, self.TEMPLATES['default'])

    async def _llm_metric(self, qid: str, query: str, expected_text: str, response_text: str,
                          *, template: Optional[str] = None) -> Optional[float]:
        """Override to use question-type-specific template"""
        if template is None:
            template = self._get_template_for_qid(qid)
        return await super()._llm_metric(qid, query, expected_text, response_text, template=template)

class KeywordMatchEvaluator(BaseEvaluator):
    """Async Evaluator for datasets using keyword matching (e.g., booksum, infbench)"""

    def __init__(self, scoring_client: OpenAI = None):
        super().__init__(scoring_client)

    def compute_keyword_hit_rate(self, prediction: str, gold_answer: str) -> float:
        """
        Compute keyword hit rate based on long_context_eval.py implementation.

        Args:
            prediction: Model's predicted answer
            gold_answer: Ground truth answer (comma-separated keywords or list)

        Returns:
            Float between 0 and 1 representing the proportion of keywords found
        """
        # Convert gold_answer to list of keywords if it's a string
        if isinstance(gold_answer, str):
            keywords = [kw.strip() for kw in gold_answer.split(",")]
        elif isinstance(gold_answer, list):
            keywords = gold_answer
        else:
            keywords = [str(gold_answer)]

        if not keywords:
            return 0.0

        # Count how many keywords appear in prediction (case-insensitive)
        prediction_lower = prediction.lower()
        hit = 0
        for keyword in keywords:
            if keyword.lower() in prediction_lower:
                hit += 1

        return hit / len(keywords)

    async def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str, existing_metrics: dict = None) -> dict:
        """Evaluate QA with keyword hit rate metric in addition to standard metrics"""
        # Get standard metrics from base evaluator (with expensive metric reuse)
        metrics = await super().evaluate_qa(qid, query, expected_answer, response, existing_metrics)

        # Add keyword hit rate metric (cheap to compute, so always recompute)
        try:
            response_text = str(response)
            expected_text = str(expected_answer)
            metrics['keyword_hit_rate'] = self.compute_keyword_hit_rate(response_text, expected_text)
        except Exception as e:
            print(f"Error computing keyword_hit_rate for qid {qid}: {e}")
            metrics['keyword_hit_rate'] = -1

        return metrics

def get_evaluator(task: str, scoring_client: OpenAI|None = None) -> BaseEvaluator:
    """Factory function to get appropriate evaluator for task"""
    if task == 'longmemeval':
        return LongmemEvalEvaluator(scoring_client)
    elif task in ['booksum', 'infbench']:
        return KeywordMatchEvaluator(scoring_client)
    else:
        return BaseEvaluator(scoring_client)

def create_client():
    """Create AsyncOpenAI client for async operations"""
    return AsyncOpenAI(**API_CONFIG_LOCAL)

def create_sync_client():
    """Create OpenAI client for sync operations"""
    return OpenAI(**API_CONFIG_LOCAL)

def create_judge_client():
    """Create AsyncOpenAI client for judging"""
    return AsyncOpenAI(**API_CONFIG_LOCAL)

def load_dataset(task: str, force_rebuild: bool = False):
    """Load dataset for given task"""
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    return DATASET_LOADERS[task](force_rebuild=force_rebuild)

async def process_single_sample(sample, agent_class, agent_kwargs, output_file, task, semaphore, write_lock):
    """Process a single sample asynchronously"""
    async with semaphore:
        print(f"{agent_class.__name__} Processing sample: {sample.task_id}")

        # Create agent instance for this sample
        agent = agent_class(**agent_kwargs)

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
                print(f"Chunk {chunk_idx} / {len(sample.chunks)} added", flush=True)
            
            # Process questions at this position
            questions_at_position = position_groups[position]
            queries = [q.query for q in questions_at_position]

            # Time the QA batch processing
            batch_start_time = time.time()
            intermediate_path = None
            tool_call_stats = None
            if agent_class.__name__ == 'VerlMemoryAgent':
                responses, intermediate_path, tool_call_stats = await agent.QA_batch_async(queries, save_intermediate=True)
            elif agent_class.__name__ == 'RLMAgent':
                responses, intermediate_path = await agent.QA_batch_async(queries, save_intermediate=True)
            else:
                responses = await agent.QA_batch_async(queries)
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
                if intermediate_path:
                    if isinstance(intermediate_path, list):
                        result['intermediate_paths'] = str(intermediate_path[i])
                    else:
                        result['intermediate_paths'] = str(intermediate_path)
                if tool_call_stats and i < len(tool_call_stats):
                    result['tool_call_stats'] = tool_call_stats[i]
                results.append(result)
        
        # Write all results to file at once (with file lock for thread safety)
        async with write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
        
        return len(results)

async def generate_responses_async(task, agent_class, agent_config, agent_id, output_dir, concurrency=32, model_name='qwen3-4b', force_overwrite=False):
    """Generate responses using specified agent with async concurrency

    Args:
        force_overwrite: If True, delete existing responses file and regenerate all responses
    """
    print(f"Starting async generation for task: {task}, agent_id: {agent_id}, model: {model_name}")
    start_time = time.time()

    # Load dataset
    eval_set = load_dataset(task)

    # Create output file
    final_output_file = f"{output_dir}/responses_{agent_id}.jsonl"

    # Check for existing partial results
    completed_qids = set()
    if force_overwrite:
        if os.path.exists(final_output_file):
            try:
                os.remove(final_output_file)
                print(f"Force overwrite enabled. Removed existing {final_output_file}")
            except OSError as exc:
                print(f"Warning: failed to remove {final_output_file}: {exc}")
        else:
            print("Force overwrite enabled but no existing responses file found to remove")
    elif os.path.exists(final_output_file):
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
    
    # Shared synchronization primitives for workers
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    
    # Create agent instances for concurrent processing
    if agent_class.__name__ == 'RLMAgent':
        client = create_sync_client()
    else:
        client = create_client()
    
    # Create tasks for all samples
    tasks = []
    for sample in tqdm.tqdm(remaining_samples, total=len(remaining_samples), desc="Scheduling samples"):
        # Prepare agent kwargs for each sample
        if agent_class.__name__ == 'FileMemoryAgent':
            agent_kwargs = {'client': client, 'task_type': task, 'task_id': sample.task_id}
        elif agent_class.__name__ == 'MemAgent':
            wo_q = agent_config.get('wo_q', False)
            agent_kwargs = {'wo_q': wo_q, 'client': client, 'model_name': model_name}
        elif agent_class.__name__ == 'MemAlphaUnifiedAgent':
            agent_kwargs = {'model_name': model_name, 'client': client, **agent_config}
        elif agent_class.__name__ == 'VerlMemoryAgent':
            agent_kwargs = {'model_name': model_name, 'client': client, 'data_source': {
                "booksum": "memalpha_booksum",
                "nlu": "memalpha_icl_nlu_8296shot_balance",
                "perltqa": "memalpha_perltqa",
                "pubmed_rct": "memalpha_pubmed-rct",
                "trec_coarse": "memalpha_icl_trec_coarse_6600shot_balance",
                "squad": "memalpha_squad",
            }.get(task, "synth" if task.startswith("synth") else task), 'agent_id': agent_id}
        else:
            # Other agents (ConcatAgent, EmergenceAgent) use model_name
            agent_kwargs = {'model_name': model_name, 'client': client}

        task_coroutine = process_single_sample(sample, agent_class, agent_kwargs, final_output_file, task, semaphore, write_lock)
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

async def evaluate_single_response(item, evaluator, results_file, semaphore, write_lock, existing_metrics_map):
    """Evaluate a single response asynchronously"""
    async with semaphore:
        qid = item['qid']
        # Get existing valid metrics for this qid (if any)
        existing_metrics = existing_metrics_map.get(qid, {})

        eval_start_time = time.time()

        metrics_info = await evaluator.evaluate_qa(
            qid=qid,
            query=item['query'],
            expected_answer=str(item['expected_answer']),
            response=str(item['response']),
            existing_metrics=existing_metrics  # Pass existing metrics to avoid recomputing
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
        if 'intermediate_paths' in item:
            result['intermediate_paths'] = item['intermediate_paths']
        if 'tool_call_stats' in item:
            result['tool_call_stats'] = item['tool_call_stats']

        # Write result to file with shared async lock
        async with write_lock:
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

        return result

async def evaluate_responses_async(input_file, task, output_dir, agent_id="unknown", concurrency=64, force_overwrite=False):
    """Evaluate responses from input file with async concurrency.

    When ``force_overwrite`` is true all existing ``evaluated_*.jsonl`` files
    under ``output_dir`` are deleted before re-running the evaluation. If an
    existing results file contains any incomplete metrics it will be fully
    re-evaluated and overwritten.
    """
    print(f"Starting async evaluation for file: {input_file}")
    start_time = time.time()

    # Load responses
    with open(input_file, 'r', encoding='utf-8') as f:
        responses = [json.loads(line) for line in f]

    # Create output file
    results_file = f"{output_dir}/evaluated_{agent_id}.jsonl"

    # Define expected metrics based on task
    # Since we always provide scoring_client in this evaluation flow, llm_score is always expected
    base_metrics = ['f1_score', 'precision', 'recall', 'exact_match', 'sub_em', 'bert_score', 'rouge_score', 'llm_score']
    if task in ['booksum', 'infbench']:
        expected_metrics = base_metrics + ['keyword_hit_rate']
    else:
        expected_metrics = base_metrics

    # Collect existing valid metrics to avoid recomputing expensive metrics
    existing_metrics_map = {}  # {qid: {metric_name: value}}

    # Check for existing partial results
    if force_overwrite:
        if Path(results_file).exists():
            try:
                Path(results_file).unlink()
                print(f"Force overwrite enabled. Removed existing {results_file}")
            except OSError as exc:
                print(f"Warning: failed to remove {results_file}: {exc}")
        else:
            print("Force overwrite enabled but no existing evaluation file found to remove")
    elif os.path.exists(results_file):
        print(f"Found existing evaluation file: {results_file}")
        needs_rerun = False
        evaluated_count = 0
        with open(results_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                evaluated_count += 1
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Line {idx}: invalid JSON ({exc}). Will re-run evaluation")
                    needs_rerun = True
                    break

                qid = result.get('qid')
                metrics = result.get('metric') or {}

                # Collect valid metrics (expensive ones: bert_score, llm_score)
                if qid:
                    valid_metrics = {k: v for k, v in metrics.items() if v != -1 and k in ['bert_score', 'llm_score']}
                    if valid_metrics:
                        existing_metrics_map[qid] = valid_metrics

                # Check if all expected metrics exist and are valid
                missing_metrics = []
                for expected_metric in expected_metrics:
                    if expected_metric not in metrics or metrics[expected_metric] == -1:
                        missing_metrics.append(expected_metric)

                if missing_metrics:
                    if not needs_rerun:
                        print(f"Found incomplete metrics for qid {qid}: missing/invalid {missing_metrics}")
                    needs_rerun = True
                    # Don't break - continue collecting valid metrics

        # Check if the number of evaluated responses matches the number of input responses
        if evaluated_count != len(responses):
            print(f"Number mismatch: {evaluated_count} evaluated vs {len(responses)} input responses. Will re-run evaluation")
            needs_rerun = True

        if needs_rerun:
            if existing_metrics_map:
                print(f"Found valid expensive metrics for {len(existing_metrics_map)} questions, will reuse them")
            # Don't delete the file anymore - we'll overwrite with new results
        else:
            print("All responses already evaluated with complete metrics. Skipping re-evaluation")
            return
    
    if not responses:
        print("No responses found for evaluation")
        return

    print(f"Evaluating {len(responses)} responses with concurrency {concurrency}")
    
    # Create evaluator and semaphore
    try:
        evaluator = get_evaluator(task, create_judge_client())
        eval_set = load_dataset(task)
        evaluator.set_category_mapping(eval_set)
        
        # Warmup BERT model in main thread before starting concurrent evaluation
        BaseEvaluator.warmup_bert_model()
    except NotImplementedError as e:
        print(f"Warning: {e}. Using default metrics.")
        return
    
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    # Create tasks for all evaluations
    tasks = [
        evaluate_single_response(item, evaluator, results_file, semaphore, write_lock, existing_metrics_map)
        for item in responses
    ]
    
    # Process all tasks concurrently with progress bar
    results = await atqdm.gather(*tasks, desc="Evaluating responses")
    
    # Calculate metrics
    metrics_acc = defaultdict(lambda: [0.0, 0])
    total_evaluation_time = sum(r['evaluation_time'] for r in results)
    
    # Aggregate metrics
    for result in results:
        metrics = result.get('metric', {})
        for metric_name, score in metrics.items():
            if score != -1:
                total, count = metrics_acc[metric_name]
                metrics_acc[metric_name] = [total + score, count + 1]
    
    # Calculate and display averages
    avg_metrics = {}
    total_time = time.time() - start_time
    avg_evaluation_time = total_evaluation_time / len(results) if results else 0
    
    print(f"Evaluation complete! Metrics:")

    for metric_name in ['f1_score', 'exact_match', 'bert_score', 'rouge_score', 'llm_score', 'keyword_hit_rate', 'precision', 'recall', 'sub_em']:
        total, valid_count = metrics_acc.get(metric_name, (0.0, 0))
        if valid_count:
            avg_score = total / valid_count
            avg_metrics[metric_name] = avg_score
            print(f"  {metric_name}: {avg_score:.4f} ({total}/{valid_count} valid)")
    
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Average evaluation time per question: {avg_evaluation_time:.2f}s")
    print(f"Results saved to: {results_file}")


def get_args():
    parser = argparse.ArgumentParser(description='Memory Agent Evaluation - Async Version')
    parser.add_argument('--task', type=str, default='longmemeval')
    parser.add_argument('--agent', type=str, default='emergence')
    parser.add_argument('--agent_id', type=str, default=None, help='Agent name for output files (default: agent type)')
    default_agent_cfg_path = (
        Path(__file__).resolve().parent
        / 'external'
        / 'memalpha'
        / 'config'
        / 'memalpha-qwen3-4b_agent_0.05-0.1.yaml'
    )
    default_agent_cfg = str(default_agent_cfg_path) if default_agent_cfg_path.exists() else None
    parser.add_argument('--agent_config_path', type=str, default=default_agent_cfg,
                        help='Optional agent configuration file (used by memalpha agent)')
    default_prompts_cfg_path = (
        Path(__file__).resolve().parent
        / 'external'
        / 'memalpha'
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
    parser.add_argument('--concurrency', type=int, default=50,
                        help='Number of concurrent tasks (default: 50)')
    parser.add_argument('--generate_only', action='store_true', help='Generate Only mode')
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Re-evaluate all responses and overwrite existing evaluated files')
    return parser.parse_args()

async def main():
    args = get_args()
    if args.agent_id is None:
        args.agent_id = args.agent
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
        print("Mode: Async Generation" if args.generate_only else "Mode: Async Generation + Evaluation")
        
        if args.agent == 'memagent_woq':
            args.agent = 'memagent'
            agent_config = {'wo_q': True}
        elif args.agent == 'memalpha':
            agent_config = {}
            if args.agent_config_path:
                agent_config['agent_config_path'] = args.agent_config_path
            if args.prompts_config_path:
                agent_config['prompts_config_path'] = args.prompts_config_path
        else:
            agent_config = {}
        # Create agent configuration
        if args.agent in AGENT_CLASS:
            agent_class = AGENT_CLASS[args.agent]
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        
        responses_file = await generate_responses_async(
            args.task, agent_class, agent_config, args.agent_id, output_dir, args.concurrency, args.model, args.force_overwrite
        )
        
        if not args.generate_only:
            await evaluate_responses_async(
                responses_file, args.task, output_dir, args.agent_id, args.concurrency,
                force_overwrite=args.force_overwrite
            )
    else:
        # Evaluation only
        print("Mode: Async Evaluation only")
        assert "responses_" in args.input_file, "Input file name must contain 'responses_'"
        agent_id = args.input_file.split("responses_")[-1].replace(".jsonl", "")
        
        await evaluate_responses_async(
            args.input_file, args.task, output_dir, agent_id, args.concurrency,
            force_overwrite=args.force_overwrite
        )

if __name__ == "__main__":
    asyncio.run(main())