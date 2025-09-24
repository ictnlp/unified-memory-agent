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
import multiprocessing as mp
from functools import partial

# Use fork as default, but force single process for emergence agent
# to avoid CUDA multiprocessing issues
import os

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
            return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
        
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
        """Calculate BERT score using CPU for emergence agent compatibility"""
        prediction = self.normalize_answer(prediction)
        ground_truth = self.normalize_answer(ground_truth)
        # Use CPU to ensure compatibility with emergence agent
        P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, 
                         rescale_with_baseline=True, device='cpu')
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
    def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        """Evaluate a single QA pair and return metrics dictionary"""
        pass

class LocomoEvaluator(BaseEvaluator):
    """Evaluator for Locomo dataset"""
    
    # LLM scoring template for Locomo
    LLM_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    
    def __init__(self, scoring_client: OpenAI):
        super().__init__()
        self.scoring_client = scoring_client
    
    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=16, max_time=300, jitter=backoff.full_jitter)
    def _llm_scoring_response(self, messages, model_name=JUDGE_MODEL_NAME):
        res = self.scoring_client.chat.completions.create(
            model=model_name, messages=messages, temperature=0, max_tokens=10
        )
        return res.choices[0].message.content
    
    def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
        try:
            category = self.qid_category_map.get(qid, 1)
            
            # F1 score calculation (original logic)
            if category == 5:  # adversarial
                f1 = 1.0 if any(phrase in response.lower() 
                               for phrase in ['no information available', 'not mentioned']) else 0.0
            elif category == 1:  # multi-hop
                f1 = f1_multi_answer(response, expected_answer)
            else:  # single-hop, temporal, open-domain
                f1 = f1_score(response, expected_answer)
            
            # Additional metrics
            exact_match = self.exact_match_score(response, expected_answer)
            bert_score_val = self.bert_score(response, expected_answer)
            rouge_score_val = self.rouge_score(response, expected_answer)
            f1_custom = self.f1_score_custom(response, expected_answer)
            
            metrics = {
                'f1_score': f1,
                'exact_match': exact_match,
                'bert_score': bert_score_val,
                'rouge_score': rouge_score_val,
                'f1_custom': f1_custom
            }
            
            # LLM score calculation (if client available)
            if self.scoring_client:
                try:
                    prompt = self.LLM_TEMPLATE.format(query, expected_answer, response)
                    messages = [{"role": "user", "content": prompt}]
                    scoring_result = self._llm_scoring_response(messages)
                    llm_score = 1 if 'yes' in scoring_result.lower() and 'no' not in scoring_result.lower() else 0
                    metrics['llm_score'] = llm_score
                except Exception as e:
                    print(f"Error in LLM scoring for qid {qid}: {e}")
                    metrics['llm_score'] = -1  # -1 indicates LLM scoring error
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating Locomo QA for qid {qid}: {e}")
            # Use -1 to indicate evaluation errors, distinguish from 0 (wrong answer)
            base_metrics = {'f1_score': -1, 'exact_match': -1, 'bert_score': -1, 'rouge_score': -1, 'f1_custom': -1}
            if self.scoring_client:
                base_metrics['llm_score'] = -1
            return base_metrics

class LongmemEvalEvaluator(BaseEvaluator):
    """Evaluator for LongmemEval dataset"""
    
    # Scoring templates
    TEMPLATES = {
        'default': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'temporal-reasoning': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'knowledge-update': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'single-session-preference': "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'abstention': "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    }
    
    def __init__(self, scoring_client: OpenAI):
        super().__init__()
        self.scoring_client = scoring_client
    
    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=16, max_time=300, jitter=backoff.full_jitter)
    def _llm_scoring_response(self, messages, model_name=JUDGE_MODEL_NAME):
        res = self.scoring_client.chat.completions.create(
            model=model_name, messages=messages, temperature=0, max_tokens=10
        )
        return res.choices[0].message.content
    
    def evaluate_qa(self, qid: str, query: str, expected_answer: str, response: str) -> dict:
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
                scoring_result = self._llm_scoring_response(messages)
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
    elif task == 'locomo':
        return LocomoEvaluator(scoring_client)
    else:
        raise NotImplementedError(f"Evaluator for task '{task}' not implemented")

def create_client():
    """Create OpenAI client with standard configuration"""
    return OpenAI(**API_CONFIG)

def create_judge_client():
    """Create OpenAI client with standard configuration"""
    return OpenAI(**API_CONFIG_LOCAL)

def load_dataset(task: str, force_rebuild: bool = False):
    """Load dataset for given task"""
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    # return DATASET_LOADERS[task](force_rebuild)[64:65] # DEBUG
    # bad_ids = json.load(open("/mnt/pfs-guan-ssai/nlu/zhangkehao/emergence_simple_fast/bad_ids.json"))
    # return [x for x in DATASET_LOADERS[task](force_rebuild) if x.questions[0].qid in bad_ids]
    return DATASET_LOADERS[task](force_rebuild)

def process_sample_subset(sample_subset, agent_class, agent_config, output_file_template, process_id, task=None):
    """Process a subset of samples in a separate process"""
    try:
        # Create agent instance for this process
        client = create_client()
        
        # Special handling for FileMemoryAgent to pass task information
        if agent_class.__name__ == 'FileMemoryAgent' and sample_subset:
            # Extract task_type and task_id from the first sample
            first_sample = sample_subset[0]
            task_type = task  # Use the task parameter passed from main
            task_id = getattr(first_sample, 'task_id', None)
            agent = agent_class(client=client, task_type=task_type, task_id=task_id)
        else:
            # Standard agent creation for other agent types
            if 'wo_q' in agent_config:
                agent = agent_class(wo_q=agent_config['wo_q'], client=client)
                if agent_config['wo_q']:
                    # Configure agent for wo_q mode
                    original_qa_batch = agent.QA_batch
                    agent.QA_batch = lambda queries: original_qa_batch(queries, wo_q=True)
            else:
                agent = agent_class(client=client)
        
        # Generate unique output file for this process
        if process_id == 0 and "{process_id}" not in output_file_template:
            # Single process mode - output_file_template is actually the final file
            output_file = output_file_template
            file_mode = 'a'  # Append mode for final file
        else:
            # Multi process mode - use template with process_id
            output_file = output_file_template.format(process_id=process_id)
            file_mode = 'w'  # Write mode for temp files
        
        # Open file for writing and keep it open throughout processing
        question_count = 0
        with open(output_file, file_mode, encoding='utf-8') as f:
            for sample_idx, sample in enumerate(sample_subset):
                print(f"Process {process_id}: Processing sample {sample_idx + 1}/{len(sample_subset)}: {sample.task_id}")
                
                # Group questions by position
                position_groups = {}
                for question in sample.questions:
                    position_groups.setdefault(question.position, []).append(question)
                
                chunk_idx = 0
                # Create progress bar for chunks (memory building)
                total_chunks = len(sample.chunks)
                chunk_pbar = tqdm.tqdm(total=total_chunks, desc=f"Building memory for {sample.task_id}", 
                                     unit="chunk", leave=False)
                
                # Process each position in order
                for position in sorted(position_groups.keys()):
                    # Add memory chunks up to current position
                    while chunk_idx <= position and chunk_idx < total_chunks:
                        agent.add_memory(sample.chunks[chunk_idx])
                        chunk_idx += 1
                        chunk_pbar.update(1)  # Update progress bar
                    
                    # Process questions at this position
                    questions_at_position = position_groups[position]
                    queries = [q.query for q in questions_at_position]
                    
                    # Time the QA batch processing
                    batch_start_time = time.time()
                    responses = agent.QA_batch(queries)
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    
                    # Write results immediately with timing info
                    for i, (q, response) in enumerate(zip(questions_at_position, responses)):
                        per_question_time = batch_duration / len(questions_at_position)
                        
                        result = {
                            'qid': q.qid,
                            'query': q.query,
                            'expected_answer': q.answer,
                            'response': response,
                            'generation_time': per_question_time
                        }
                        # 实时写入文件并刷新缓冲区
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()  # 确保立即写入磁盘
                        question_count += 1
                        
                        print(f"Process {process_id}: Completed question {question_count} - {q.qid}")
                
                # Close the progress bar after processing all positions
                chunk_pbar.close()
        
        return question_count, output_file
        
    except Exception as e:
        print(f"Error in process {process_id}: {e}")
        # 如果出错，仍然返回已创建的文件路径（可能包含部分结果）
        output_file = output_file_template.format(process_id=process_id)
        if os.path.exists(output_file):
            # 统计已写入的行数
            with open(output_file, 'r', encoding='utf-8') as f:
                completed_count = sum(1 for line in f if line.strip())
            return completed_count, output_file
        return 0, None

def generate_responses(task, agent_or_config, agent_id, output_dir, num_processes=None):
    """Generate responses using specified agent with multiprocessing"""
    print(f"Starting generation for task: {task}, agent_id: {agent_id}")
    start_time = time.time()
    
    # Force single process for emergence agent to avoid spawn issues
    if 'emergence' in agent_id.lower():
        print("Emergence agent detected, using single process mode for stability")
        num_processes = 1
    
    # Handle both agent instance and (agent_class, agent_config) tuple
    if isinstance(agent_or_config, tuple):
        agent_class, agent_config = agent_or_config
    else:
        # Legacy: agent instance passed
        agent = agent_or_config
        agent_config = get_agent_config(agent)
        agent_class = type(agent)
    
    # Load dataset
    eval_set = load_dataset(task)

    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)  # Use up to 4 processes by default
    
    # Adjust num_processes if we have fewer samples
    num_processes = min(num_processes, len(eval_set))
    
    print(f"Using {num_processes} processes for parallel generation")
    
    # Create output file (without timestamp for resumability)
    final_output_file = f"{output_dir}/responses_{agent_id}.jsonl"
    
    # Check for existing partial results and determine what's already done
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
    
    # Filter out completed samples and questions
    remaining_samples = []
    for sample in eval_set:
        remaining_questions = [q for q in sample.questions if q.qid not in completed_qids]
        if remaining_questions:
            # Create a new sample with only remaining questions
            remaining_sample = type(sample)(
                task_id=sample.task_id,
                chunks=sample.chunks,
                questions=remaining_questions
            )
            remaining_samples.append(remaining_sample)
    if not remaining_samples:
        print("All questions already completed!")
        return final_output_file
    
    print(f"Processing {len(remaining_samples)} remaining samples")
    
    if num_processes == 1:
        # Sequential processing - write directly to final file
        results = [process_sample_subset(remaining_samples, agent_class, 
                                       agent_config, 
                                       final_output_file, 0, task)]
    else:
        # Split dataset into chunks for each process
        chunk_size = len(remaining_samples) // num_processes
        sample_chunks = []
        
        for i in range(num_processes):
            start_idx = i * chunk_size
            if i == num_processes - 1:  # Last chunk gets remaining samples
                end_idx = len(remaining_samples)
            else:
                end_idx = (i + 1) * chunk_size
            
            if start_idx < len(remaining_samples):
                sample_chunks.append(remaining_samples[start_idx:end_idx])
        
        # Create output file template
        output_file_template = f"{output_dir}/temp_{{process_id}}.jsonl"
        
        # Process chunks in parallel
        with mp.Pool(processes=len(sample_chunks)) as pool:
            # Map each chunk to a process with correct argument order
            process_args = [(chunk, agent_class, agent_config, output_file_template, i, task) 
                           for i, chunk in enumerate(sample_chunks)]
            results = pool.starmap(process_sample_subset, process_args)
    
    # Combine results from all processes
    total_processed = 0
    process_files = []
    
    for questions_count, output_file in results:
        process_files.append(output_file)
        if output_file and questions_count > 0:
            total_processed += questions_count
        elif output_file is None:
            print(f"Warning: One process failed to generate results")
    
    # For single process mode, results are already in final file, skip combining
    if num_processes == 1:
        # Single process mode - no need to combine files
        pass
    else:
        # Append new results to existing file for multi-process mode
        with open(final_output_file, 'a', encoding='utf-8') as final_f:
            for process_file in process_files:
                if os.path.exists(process_file):
                    with open(process_file, 'r', encoding='utf-8') as pf:
                        for line in pf:
                            final_f.write(line)
                    # Clean up process-specific file
                    os.remove(process_file)
    
    total_time = time.time() - start_time
    print(f"Generation complete! Total time: {total_time:.2f}s")
    print(f"Processed {total_processed} new questions across {len(process_files)} processes")
    print(f"Average time per question: {total_time/total_processed:.2f}s" if total_processed > 0 else "No new questions processed")
    print(f"Results saved to: {final_output_file}")
    
    return final_output_file

def get_agent_config(agent):
    """Extract agent configuration for multiprocessing"""
    agent_config = {}
    if hasattr(agent, 'wo_q'):
        agent_config['wo_q'] = getattr(agent, 'wo_q', False)
    return agent_config

def process_evaluation_subset(response_subset, task, output_file_template, process_id):
    """Process a subset of responses for evaluation in a separate process"""
    try:
        # Create evaluator for this process
        try:
            evaluator = get_evaluator(task, create_judge_client())
            eval_set = load_dataset(task)
            evaluator.set_category_mapping(eval_set)
        except NotImplementedError as e:
            print(f"Process {process_id} Warning: {e}. Using default metrics.")
            evaluator = None
        
        # Generate unique output file for this process
        if process_id == 0 and "{process_id}" not in output_file_template:
            # Single process mode - output_file_template is actually the final file
            output_file = output_file_template
            file_mode = 'a'  # Append mode for final file
        else:
            # Multi process mode - use template with process_id
            output_file = output_file_template.format(process_id=process_id)
            file_mode = 'w'  # Write mode for temp files
        
        # Open file for writing and keep it open throughout processing
        evaluation_count = 0
        with open(output_file, file_mode, encoding='utf-8') as f:
            for item in response_subset:
                if evaluator:
                    # Time the evaluation
                    eval_start_time = time.time()
                    metrics_info = evaluator.evaluate_qa(
                        qid=item['qid'], query=item['query'],
                        expected_answer=item['expected_answer'], response=item['response']
                    )
                    eval_end_time = time.time()
                    evaluation_time = eval_end_time - eval_start_time
                    
                    metrics = {k: v for k, v in metrics_info.items() if isinstance(v, (int, float))}
                else:
                    evaluation_time = 0
                    metrics = {'default_score': 0}
                
                result = {
                    'qid': item['qid'],
                    'query': item['query'],
                    'expected_answer': item['expected_answer'],
                    'response': item['response'],
                    'metric': metrics,
                    'evaluation_time': evaluation_time
                }
                
                # Add generation_time if it exists in the original item
                if 'generation_time' in item:
                    result['generation_time'] = item['generation_time']
                
                # 实时写入文件并刷新缓冲区
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # 确保立即写入磁盘
                evaluation_count += 1
                
                print(f"Process {process_id}: Completed evaluation {evaluation_count} - {item['qid']}")
        
        return evaluation_count, output_file
        
    except Exception as e:
        print(f"Error in evaluation process {process_id}: {e}")
        # 如果出错，仍然返回已创建的文件路径（可能包含部分结果）
        output_file = output_file_template.format(process_id=process_id)
        if os.path.exists(output_file):
            # 统计已写入的行数
            with open(output_file, 'r', encoding='utf-8') as f:
                completed_count = sum(1 for line in f if line.strip())
            return completed_count, output_file
        return 0, None

def evaluate_responses(input_file, task, output_dir, agent_id="unknown", num_processes=None):
    """Evaluate responses from input file with multiprocessing"""
    print(f"Starting evaluation for file: {input_file}")
    start_time = time.time()
    
    # Force single process for emergence agent to avoid spawn issues
    if 'emergence' in agent_id.lower():
        print("Emergence agent detected, using single process mode for stability")
        num_processes = 1
    
    # Load responses
    with open(input_file, 'r', encoding='utf-8') as f:
        responses = [json.loads(line) for line in f]
    
    # Create output file (without timestamp for resumability)
    results_file = f"{output_dir}/evaluated_{agent_id}.jsonl"
    
    # Check for existing partial results and determine what's already done
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
    
    print(f"Evaluating {len(remaining_responses)} remaining responses")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)  # Use up to 4 processes by default
    
    # Adjust num_processes if we have fewer responses
    num_processes = min(num_processes, len(remaining_responses))
    
    print(f"Using {num_processes} processes for parallel evaluation")
    
    if num_processes == 1:
        # Sequential processing - write directly to final file
        results = [process_evaluation_subset(remaining_responses, task, 
                                           results_file, 0)]
    else:
        # Split responses into chunks for each process
        chunk_size = len(remaining_responses) // num_processes
        response_chunks = []
        
        for i in range(num_processes):
            start_idx = i * chunk_size
            if i == num_processes - 1:  # Last chunk gets remaining responses
                end_idx = len(remaining_responses)
            else:
                end_idx = (i + 1) * chunk_size
            
            if start_idx < len(remaining_responses):
                response_chunks.append(remaining_responses[start_idx:end_idx])
        
        # Create output file template
        output_file_template = f"{output_dir}/eval_temp_{{process_id}}.jsonl"
        
        # Process chunks in parallel
        with mp.Pool(processes=len(response_chunks)) as pool:
            # Map each chunk to a process with correct argument order
            process_args = [(chunk, task, output_file_template, i) 
                           for i, chunk in enumerate(response_chunks)]
            results = pool.starmap(process_evaluation_subset, process_args)
    
    # Combine results from all processes
    total_processed = 0
    process_files = []
    
    for evaluation_count, output_file in results:
        process_files.append(output_file)
        if output_file and evaluation_count > 0:
            total_processed += evaluation_count
        elif output_file is None:
            print(f"Warning: One evaluation process failed")
    
    # For single process mode, results are already in final file, skip combining
    if num_processes == 1:
        # Single process mode - no need to combine files
        pass
    else:
        # Append new results to existing file for multi-process mode
        with open(results_file, 'a', encoding='utf-8') as final_f:
            for process_file in process_files:
                if os.path.exists(process_file):
                    with open(process_file, 'r', encoding='utf-8') as pf:
                        for line in pf:
                            final_f.write(line)
                    # Clean up process-specific file
                    os.remove(process_file)
    
    # Calculate overall metrics from the final file
    total_metrics = {}
    total_evaluation_time = 0
    all_results = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                result = json.loads(line)
                all_results.append(result)
                if 'evaluation_time' in result:
                    total_evaluation_time += result['evaluation_time']
            except:
                continue
    
    # Aggregate metrics from all results
    for result in all_results:
        metrics = result.get('metric', {})
        for metric_name, score in metrics.items():
            if score != -1:  # Only aggregate valid scores
                total_metrics[metric_name] = total_metrics.get(metric_name, 0) + score
                # Track valid count per metric
                if f"{metric_name}_count" not in total_metrics:
                    total_metrics[f"{metric_name}_count"] = 0
                total_metrics[f"{metric_name}_count"] += 1
            else:
                # Track error count per metric
                if f"{metric_name}_errors" not in total_metrics:
                    total_metrics[f"{metric_name}_errors"] = 0
                total_metrics[f"{metric_name}_errors"] += 1
    
    # Calculate and display averages
    avg_metrics = {}
    total_time = time.time() - start_time
    avg_evaluation_time = total_evaluation_time / len(all_results) if all_results else 0
    
    print(f"Evaluation complete! Metrics:")
    
    for metric_name in ['f1_score', 'exact_match', 'bert_score', 'rouge_score', 'f1_custom', 'llm_score']:
        if metric_name in total_metrics:
            valid_count = total_metrics.get(f"{metric_name}_count", 0)
            error_count = total_metrics.get(f"{metric_name}_errors", 0)
            
            if valid_count > 0:
                avg_score = total_metrics[metric_name] / valid_count
                avg_metrics[metric_name] = avg_score
                print(f"  {metric_name}: {avg_score:.4f} ({total_metrics[metric_name]}/{valid_count} valid", end="")
                if error_count > 0:
                    print(f", {error_count} errors)", end="")
                print(")")
            else:
                avg_metrics[metric_name] = 0
                print(f"  {metric_name}: No valid scores ({error_count} errors)")
    
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Processed {total_processed} new evaluations across {len(process_files)} processes")
    print(f"Average evaluation time per question: {avg_evaluation_time:.2f}s")
    print(f"Results saved to: {results_file}")
    
    return results_file, avg_metrics

def get_args():
    parser = argparse.ArgumentParser(description='Memory Agent Evaluation')
    parser.add_argument('--task', type=str, choices=['locomo', 'longmemeval'], default='locomo')
    parser.add_argument('--agent', type=str, choices=['concat', 'memagent', 'filememory', 'emergence'], default='concat')
    parser.add_argument('--wo_q', action='store_true', help='Use wo_q mode for MemAgent')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file for evaluation only (JSONL format)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/{task})')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of processes for parallel generation (default: min(cpu_count, 4))')
    parser.add_argument('--generate_only', action='store_true', help='Generate Only mode')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Setup output directory
    output_dir = args.output_dir or f"results/{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.input_file is None:
        # Generation + Evaluation
        print("Mode: Generation + Evaluation")
        
        # Create agent with appropriate configuration
        if args.agent in AGENT_CLASS:
            # Determine agent_id based on configuration
            if args.agent == 'memagent' and args.wo_q:
                # For multiprocessing, we only need the agent class and config
                agent_class = AGENT_CLASS[args.agent]
                agent_config = {'wo_q': True}
                agent_id = 'memagent_woq'
            elif args.agent == 'filememory':
                # For FileMemoryAgent, don't create instance here since it needs task info
                agent_class = AGENT_CLASS[args.agent]
                agent_config = {}
                agent_id = args.agent
            else:
                agent_id = args.agent
                agent_class = AGENT_CLASS[args.agent]
                agent_config = {}
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        
        responses_file = generate_responses(args.task, (agent_class, agent_config), agent_id, output_dir, args.num_processes)
        if not args.generate_only:
            evaluate_responses(responses_file, args.task, output_dir, agent_id, args.num_processes)
    else:
        # Evaluation only
        print("Mode: Evaluation only")
        # Extract agent_id from filename
        agent_id = "unknown"
        if "responses_" in args.input_file:
            try:
                # Handle both regular and woq naming
                filename_part = args.input_file.split("responses_")[1]
                if "_woq_" in filename_part:
                    agent_id = filename_part.split("_woq_")[0] + "_woq"
                else:
                    agent_id = filename_part.split("_")[0]
            except:
                pass
        evaluate_responses(args.input_file, args.task, output_dir, agent_id, args.num_processes)

if __name__ == "__main__":
    main()