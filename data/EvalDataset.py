import json
import os
from typing import Callable
from pydantic import BaseModel
import random
import pandas as pd

CONV_START_PROMPT = "Below is a conversation between {} and {}.\n\n"

# Data paths - can be overridden by environment variables
# Default paths point to original Mem-alpha data location
# On new machines, set MEMALPHA_DATA_DIR environment variable or copy data files
MEMALPHA_PARQUET_PATH = os.getenv(
    "MEMALPHA_PARQUET_PATH",
    "./data/raw/memalpha/test.parquet"
)
MEMORYAGENTBENCH_PARQUET_PATH = os.getenv(
    "MEMORYAGENTBENCH_PARQUET_PATH",
    "./data/raw/memoryagentbench/test.parquet"
)

BenchmarkRegistry = dict[str, Callable[..., list["EvalData"]]]
BENCHMARK_REGISTRY: BenchmarkRegistry = {}

# CLASSIFICATION_TASK_HINT = (
#     "Task: This is a classification sample. Respond only with the correct label "
#     "for the sentence below rather than answering the sentence itself."
# )

CLASSIFICATION_TASK_HINT = "What are the labels for the above sentence?"

def _apppend_classification_hint(samples: list["EvalData"]) -> bool:
    """Prepend explicit classification instruction to each question query."""
    modified = False
    for sample in samples:
        for question in sample.questions:
            if CLASSIFICATION_TASK_HINT in question.query:
                continue
            question.query = f"Sentence: {question.query}\n{CLASSIFICATION_TASK_HINT}"
            modified = True
    return modified


def register_benchmark(name: str | None = None) -> Callable[[Callable[..., list["EvalData"]]], Callable[..., list["EvalData"]]]:
    """Decorator to register benchmark loaders for easy discovery."""
    def decorator(func: Callable[..., list["EvalData"]]) -> Callable[..., list["EvalData"]]:
        key = name or func.__name__.removeprefix("load_")
        if key in BENCHMARK_REGISTRY:
            raise KeyError(f"Benchmark '{key}' is registered multiple times.")
        BENCHMARK_REGISTRY[key] = func
        return func

    return decorator

random.seed(114514)

class Question(BaseModel):
    qid: str|None = None
    query: str
    answer: str
    position: int
    category: str|int|None = None  # For Locomo (int) and LongmemEval (str)

class EvalData(BaseModel):
    task_id: str
    questions: list[Question]
    chunks: list[str]


def _parse_chunks(raw_chunks) -> list[str]:
    if isinstance(raw_chunks, str):
        try:
            chunk_entries = json.loads(raw_chunks)
        except json.JSONDecodeError:
            chunk_entries = []
    elif isinstance(raw_chunks, (list, tuple)):
        chunk_entries = list(raw_chunks)
    else:
        chunk_entries = []

    chunks: list[str] = []
    for chunk in chunk_entries:
        if isinstance(chunk, str):
            chunks.append(chunk)
        else:
            chunks.append(json.dumps(chunk, ensure_ascii=False))
    return chunks


def _parse_qa_pairs(raw_field) -> list:
    if isinstance(raw_field, str):
        try:
            return json.loads(raw_field)
        except json.JSONDecodeError:
            return []
    if isinstance(raw_field, (list, tuple)):
        return list(raw_field)
    return []


def _resolve_identifier(row, candidates: list[str], fallback: str) -> str:
    for key in candidates:
        if key is None:
            continue
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return fallback


def _extract_question_answer(qa_pair) -> tuple[str, str]:
    question_text = ""
    answer_value: str | list | tuple | None = ""

    if isinstance(qa_pair, dict):
        question_text = qa_pair.get("question", "")
        answer_value = qa_pair.get("answer", qa_pair.get("answers", ""))
    elif isinstance(qa_pair, (list, tuple)):
        if len(qa_pair) >= 3:
            question_text = qa_pair[1]
            answer_value = qa_pair[2]
        elif len(qa_pair) >= 2:
            question_text = qa_pair[0]
            answer_value = qa_pair[1]

    if isinstance(answer_value, (list, tuple)):
        answer_text = ", ".join(str(x) for x in answer_value)
    else:
        answer_text = str(answer_value or "")

    return str(question_text or ""), answer_text


def _build_evaldata(df: pd.DataFrame, *, id_keys: list[str], task_prefix: str, default_source: str) -> list[EvalData]:
    processed: list[EvalData] = []

    for row_idx, row in df.iterrows():
        chunks = _parse_chunks(row.get("chunks"))
        qa_pairs = _parse_qa_pairs(row.get("questions_and_answers"))

        base_id = _resolve_identifier(row, id_keys, fallback=str(row_idx))
        task_id = f"{task_prefix}_{base_id}"

        data_source_raw = row.get("data_source", default_source)
        if data_source_raw is None or (isinstance(data_source_raw, float) and pd.isna(data_source_raw)):
            data_source_raw = default_source
        data_source = str(data_source_raw)

        final_position = max(len(chunks) - 1, 0)
        questions: list[Question] = []

        for q_idx, qa_pair in enumerate(qa_pairs):
            question_text, answer_text = _extract_question_answer(qa_pair)
            if not question_text:
                continue

            questions.append(Question(
                qid=f"{task_id}_{q_idx}",
                query=question_text,
                answer=answer_text,
                position=final_position,
                category=data_source
            ))

        processed.append(EvalData(
            task_id=task_id,
            questions=questions,
            chunks=chunks
        ))

    return processed


def _load_parquet_subset(
    *,
    parquet_path: str,
    cache_path: str,
    data_source_value: str,
    task_prefix: str,
    default_source: str,
    id_keys: list[str],
    force_rebuild: bool,
) -> list[EvalData]:
    if os.path.exists(cache_path) and not force_rebuild:
        return load_from_path(cache_path)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Source parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    subset = df[df["data_source"] == data_source_value]
    if subset.empty:
        raise ValueError(f"No entries found for data_source '{data_source_value}'")

    processed = _build_evaldata(
        subset,
        id_keys=id_keys,
        task_prefix=task_prefix,
        default_source=default_source
    )

    json.dump([item.model_dump() for item in processed], open(cache_path, "w"), indent=4, ensure_ascii=False)
    return processed

@register_benchmark()
def load_locomo(force_rebuild=False):
    # 有五类QA
    # 1. multi hop
    # 2. temporal
    # 3. open domain
    # 4. single hop
    # 5. adversarial
    file_path = "data/processed_locomo.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)
    raw = json.load(open("data/raw/locomo10.json"))
    processed = []
    for item in raw:
        questions = []
        num_sessions = len(item['session_summary'])
        for idx, qa in enumerate(item['qa']):
            if qa['evidence']:
                try:
                    last_sess_id = max([
                        int(ev[1:].split(":")[0]) for ev in qa['evidence']
                    ])
                except:
                    breakpoint()
            else:
                last_sess_id = num_sessions
            # sess_position = random.randint(last_sess_id, num_sessions)
            sess_position = num_sessions # 选择最后一个位置提问
            if qa['category'] == 2:
                question = qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.'
                answer = qa['answer']
            elif qa['category'] == 5:
                question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', qa['adversarial_answer'])
                    answer = 'a'
                else:
                    question = question.format(qa['adversarial_answer'], 'Not mentioned in the conversation')
                    answer = 'b'
            else:
                question = qa['question']
                answer = qa['answer']
            questions.append(Question(
                qid=f'locomo_{item['sample_id']}_{idx}',
                query=question,
                answer=str(answer),
                position=sess_position - 1, # 下标从零开始
                category=qa['category']  # Add category from original data
            ))
        questions = sorted(questions, key=lambda x:x.position)
        chunks = []
        session = item['conversation']
        speaker_a = session['speaker_a']
        speaker_b = session['speaker_b']
        start_prompt = CONV_START_PROMPT.format(speaker_a, speaker_b)
        session_nums = [int(k.split('_')[-1]) for k in session.keys() if 'session' in k and 'date_time' not in k]
        for session_num in sorted(session_nums):
            if f'session_{session_num}' in session:
                session_time = session[f'session_{session_num}_date_time']
                session_conv = ''
                
                for dialog in session[f'session_{session_num}']:
                    turn = dialog['speaker'] + ' said, "' + dialog['text'] + '"'
                    if 'blip_caption' in dialog:
                        turn += ' and shared ' + dialog['blip_caption']
                    turn += '\n'
                    session_conv += turn
                query_conv = 'DATE: ' + session_time + '\n' + 'CONVERSATION:\n' + session_conv + '\n\n'
                chunks.append(start_prompt + query_conv)

        evaldata = EvalData(task_id=f"locomo_{item['sample_id']}", questions=questions, chunks=chunks)
        processed.append(evaldata)
    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed

@register_benchmark()
def load_longmemeval(force_rebuild=False, oracle=False):
    file_path = "data/processed_longmemeval_oracle.json" if oracle else "data/processed_longmemeval.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)
    
    raw = json.load(open("data/raw/longmemeval_s.json"))
    processed = []
    
    for item in raw:
        task_id = item['task_id']
        questions = []
        
        # Process questions
        for idx, q_data in enumerate(item['questions']):
            # Add question time to the question text
            question_text = f"[{q_data['question_time']}] {q_data['question']}"
            
            # Find position based on has_answer evidence
            position = 0  # Default to asking at the beginning
            
            # Find the last session that contains evidence (has_answer=True)
            for session_idx, session in enumerate(item['sessions']):
                for turn in session:
                    if turn.get('has_answer', False):
                        position = session_idx
            
            # position = random.randint(position + 1, len(item['sessions'])) # 随机选择最后一个证据到最后一个turn之间的某个位置
            position = len(item['sessions']) #选择最后一个位置提问
            
            questions.append(Question(
                qid=f"longmemeval_{task_id}_{idx}",
                query=question_text,
                answer=str(q_data['correct_answer']),
                position=position - 1,  # Convert to 0-based index
                category=q_data['question_type']  # Add question_type as category
            ))
        
        # Process sessions into chunks
        chunks = []
        start_prompt = CONV_START_PROMPT.format("user", "assistant")
        
        for session_idx, session in enumerate(item['sessions']):
            if not session:  # Skip empty sessions
                continue
                
            # Get date from first turn in session (all turns in same session have same date)
            session_date = session[0]['date']
            session_conv = ''
            
            for turn in session:
                if oracle and not turn.get('has_answer', False):
                    continue
                role = "User" if turn['role'] == 'user' else "Assistant"
                turn_text = f'{role} said, "{turn["content"]}"'
                turn_text += '\n'
                session_conv += turn_text
            if not session_conv:
                session_conv = "NO CONVERSATION"
            query_conv = f'DATE: {session_date}\nCONVERSATION:\n{session_conv}\n\n'
            chunks.append(start_prompt + query_conv)
        
        evaldata = EvalData(
            task_id=f"longmemeval_{task_id}", 
            questions=questions, 
            chunks=chunks
        )
        processed.append(evaldata)
    
    # Save processed data
    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed

def load_hotpotqa_deprecated(force_rebuild=False, num_docs: int=200, num_queries: int=5, num_samples: int=100):
    """Deprecated: Old HotpotQA loader using HuggingFace dataset"""
    file_path = f"data/processed_hotpotqa_{num_docs}_{num_queries}_{num_samples}.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)
    
    from datasets import load_dataset
    raw = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    # Extract all documents and create a mapping
    all_docs = []
    doc_to_index = {}
    
    for item in raw:
        # HotpotQA context format: {'title': [title1, title2, ...], 'sentences': [[sent1, sent2], [sent3, sent4], ...]}
        titles = item['context']['title']
        sentences = item['context']['sentences']
        
        for title, sentence_list in zip(titles, sentences):
            doc_text = f"{title}\n{''.join(sentence_list)}"
            if doc_text not in doc_to_index:
                doc_to_index[doc_text] = len(all_docs)
                all_docs.append(doc_text)
    
    processed = []
    random.seed(42)  # Ensure reproducible results
    
    # Limit to num_samples items
    sample_indices = random.sample(range(len(raw)), min(num_samples, len(raw)))
    
    for sample_idx in sample_indices:
        item = raw[sample_idx]
        task_id = f"hotpotqa_{item['id']}"
        
        # Find supporting documents based on supporting_facts
        supporting_doc_indices = set()
        supporting_titles = item['supporting_facts']['title']
        
        for title in supporting_titles:
            # Find the document with this title
            for doc_text in all_docs:
                if doc_text.startswith(title + "\n"):
                    supporting_doc_indices.add(doc_to_index[doc_text])
                    break
        
        # Select additional random documents to reach num_docs total
        available_indices = [i for i in range(len(all_docs)) if i not in supporting_doc_indices]
        additional_needed = max(0, num_docs - len(supporting_doc_indices))
        additional_indices = random.sample(available_indices, min(additional_needed, len(available_indices)))
        
        # Combine supporting and additional documents
        selected_doc_indices = list(supporting_doc_indices) + additional_indices
        selected_docs = [all_docs[i] for i in selected_doc_indices]
        
        # Shuffle documents to randomize order
        random.shuffle(selected_docs)
        
        # Create chunks from documents with proper formatting
        chunks = []
        DOCUMENT_PROMPT = "Document {i}:\n{document}"
        
        for i, doc in enumerate(selected_docs):
            formatted_doc = DOCUMENT_PROMPT.format(i=i+1, document=doc)
            chunks.append(formatted_doc)
        
        # Generate multiple questions at different positions
        questions = []
        
        # Find the latest position where all supporting evidence is available
        # Since we don't have a temporal order in HotpotQA, we'll ensure supporting docs
        # are in the first part of chunks
        supporting_docs_in_chunks = []
        for chunk_idx, chunk in enumerate(chunks):
            for supporting_idx in supporting_doc_indices:
                if all_docs[supporting_idx] in chunk:
                    supporting_docs_in_chunks.append(chunk_idx)
        
        # The latest position where all evidence is available
        if supporting_docs_in_chunks:
            latest_evidence_position = max(supporting_docs_in_chunks)
        else:
            # Fallback: place supporting docs in first few chunks
            latest_evidence_position = min(len(supporting_doc_indices), len(chunks)) - 1
        
        # Generate num_queries questions at different positions
        for q_idx in range(num_queries):
            # Question can be asked from the latest evidence position onwards
            min_position = latest_evidence_position
            max_position = len(chunks) - 1
            
            if min_position <= max_position:
                # Distribute questions across valid positions
                if num_queries == 1:
                    position = max_position  # Ask at the end if only one question
                else:
                    # Distribute evenly across valid range
                    position_range = max_position - min_position
                    position = min_position + (q_idx * position_range) // (num_queries - 1)
            else:
                position = len(chunks) - 1  # Fallback to last position
            
            questions.append(Question(
                qid=f"hotpotqa_{item['id']}_{q_idx}",
                query=item['question'],
                answer=item['answer'],
                position=position,
                category='multi-hop'  # HotpotQA is multi-hop reasoning
            ))
        
        # Sort questions by position
        questions = sorted(questions, key=lambda x: x.position)
        
        evaldata = EvalData(
            task_id=task_id,
            questions=questions,
            chunks=chunks
        )
        processed.append(evaldata)
    
    # Save processed data
    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed

@register_benchmark()
def load_hotpotqa(force_rebuild=False, num_docs: int=200, num_queries: int=1, num_samples: int=128):
    """
    Load HotpotQA data from MemAgent_minimal format
    
    Args:
        force_rebuild: Force rebuild cached data
        num_docs: Number of documents (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200)
        num_queries: Number of queries per sample (fixed at 1 for this format)
        num_samples: Number of samples (fixed at 128 for this format)
    """
    # Fixed values
    num_queries = 1
    num_samples = 128
    
    file_path = f"data/processed_hotpotqa_{num_docs}.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)
    
    # Load source data
    source_file = f"./data/raw/hotpotqa/eval_{num_docs}.json"
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}. Valid num_docs: 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200")
    
    raw = json.load(open(source_file))
    processed = []
    
    for item in raw:
        # Extract fields
        context = item['context']  # Full context with all documents
        question = item['input']
        answers = item['answers']  # List of valid answers
        index = item['index']
        actual_num_docs = item['num_docs']
        
        # Parse context into chunks (split by "Document N:")
        chunks = []
        doc_parts = context.split('\nDocument ')
        
        # First part is "Document 1:..."
        if doc_parts[0].startswith('Document '):
            chunks.append(doc_parts[0].strip())
        
        # Rest are "N:..."
        for part in doc_parts[1:]:
            chunks.append(f'Document {part.strip()}')
        
        # Create single question at the end (after all documents)
        questions = [
            Question(
                qid=f"hotpotqa_{num_docs}_{index}",
                query=question,
                answer=answers[0] if answers else "",  # Use first answer
                position=len(chunks) - 1,  # Ask at the end
                category='qa'
            )
        ]
        
        evaldata = EvalData(
            task_id=f"hotpotqa_{num_docs}_{index}",
            questions=questions,
            chunks=chunks
        )
        processed.append(evaldata)
    
    # Save processed data
    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed


@register_benchmark()
def load_memalpha(force_rebuild: bool = False) -> list[EvalData]:
    """Load MemAlpha benchmark data into EvalData format."""
    file_path = "data/processed_memalpha.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)

    if not os.path.exists(MEMALPHA_PARQUET_PATH):
        raise FileNotFoundError(f"MemAlpha source file not found: {MEMALPHA_PARQUET_PATH}")

    df = pd.read_parquet(MEMALPHA_PARQUET_PATH)
    processed = _build_evaldata(
        df,
        id_keys=["instance_id"],
        task_prefix="memalpha",
        default_source="memalpha"
    )

    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed


@register_benchmark()
def load_trec_coarse(force_rebuild: bool = False) -> list[EvalData]:
    """Load TREC-Coarse subset from MemoryAgentBench into EvalData format."""
    cache_path = "data/processed_trec_coarse.json"
    data = _load_parquet_subset(
        parquet_path=MEMORYAGENTBENCH_PARQUET_PATH,
        cache_path=cache_path,
        data_source_value="icl_trec_coarse_6600shot_balance",
        task_prefix="trec_coarse",
        default_source="trec_coarse",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )
    if _apppend_classification_hint(data):
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump([item.model_dump() for item in data], cache_file, indent=4, ensure_ascii=False)
    return data


@register_benchmark()
def load_banking77(force_rebuild: bool = False) -> list[EvalData]:
    """Load Banking77 subset from MemoryAgentBench into EvalData format."""
    cache_path = "data/processed_banking77.json"
    data = _load_parquet_subset(
        parquet_path=MEMORYAGENTBENCH_PARQUET_PATH,
        cache_path=cache_path,
        data_source_value="icl_banking77_5900shot_balance",
        task_prefix="banking77",
        default_source="banking77",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )
    if _apppend_classification_hint(data):
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump([item.model_dump() for item in data], cache_file, indent=4, ensure_ascii=False)
    return data


@register_benchmark()
def load_clinic(force_rebuild: bool = False) -> list[EvalData]:
    """Load Clinic150 subset from MemoryAgentBench into EvalData format."""
    cache_path = "data/processed_clinic.json"
    data = _load_parquet_subset(
        parquet_path=MEMORYAGENTBENCH_PARQUET_PATH,
        cache_path=cache_path,
        data_source_value="icl_clinic150_7050shot_balance",
        task_prefix="clinic",
        default_source="clinic",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )
    if _apppend_classification_hint(data):
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump([item.model_dump() for item in data], cache_file, indent=4, ensure_ascii=False)
    return data


@register_benchmark()
def load_nlu(force_rebuild: bool = False) -> list[EvalData]:
    """Load NLU subset from MemoryAgentBench into EvalData format."""
    cache_path = "data/processed_nlu.json"
    data = _load_parquet_subset(
        parquet_path=MEMORYAGENTBENCH_PARQUET_PATH,
        cache_path=cache_path,
        data_source_value="icl_nlu_8296shot_balance",
        task_prefix="nlu",
        default_source="nlu",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )
    if _apppend_classification_hint(data):
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump([item.model_dump() for item in data], cache_file, indent=4, ensure_ascii=False)
    return data


@register_benchmark()
def load_trec_fine(force_rebuild: bool = False) -> list[EvalData]:
    """Load TREC-Fine subset from MemoryAgentBench into EvalData format."""
    cache_path = "data/processed_trec_fine.json"
    data = _load_parquet_subset(
        parquet_path=MEMORYAGENTBENCH_PARQUET_PATH,
        cache_path=cache_path,
        data_source_value="icl_trec_fine_6400shot_balance",
        task_prefix="trec_fine",
        default_source="trec_fine",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )
    if _apppend_classification_hint(data):
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump([item.model_dump() for item in data], cache_file, indent=4, ensure_ascii=False)
    return data

@register_benchmark()
def load_infbench(force_rebuild: bool = False) -> list[EvalData]:
    """Load infbench subset from MemoryAgentBench into EvalData format."""
    cache_path = "data/processed_infbench.json"
    data = _load_parquet_subset(
        parquet_path=MEMORYAGENTBENCH_PARQUET_PATH,
        cache_path=cache_path,
        data_source_value="infbench_sum_eng_shots2",
        task_prefix="infbench",
        default_source="infbench",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )
    return data

@register_benchmark()
def load_booksum(force_rebuild: bool = False) -> list[EvalData]:
    """Load BookSum subset from MemAlpha into EvalData format."""
    return _load_parquet_subset(
        parquet_path=MEMALPHA_PARQUET_PATH,
        cache_path="data/processed_booksum.json",
        data_source_value="booksum",
        task_prefix="booksum",
        default_source="booksum",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )


@register_benchmark()
def load_perltqa(force_rebuild: bool = False) -> list[EvalData]:
    """Load PerLTQA subset from MemAlpha into EvalData format."""
    return _load_parquet_subset(
        parquet_path=MEMALPHA_PARQUET_PATH,
        cache_path="data/processed_perltqa.json",
        data_source_value="perltqa",
        task_prefix="perltqa",
        default_source="perltqa",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )


@register_benchmark()
def load_pubmed_rct(force_rebuild: bool = False) -> list[EvalData]:
    """Load PubMed-RCT subset from MemAlpha into EvalData format."""
    return _load_parquet_subset(
        parquet_path=MEMALPHA_PARQUET_PATH,
        cache_path="data/processed_pubmed_rct.json",
        data_source_value="pubmed-rct",
        task_prefix="pubmed_rct",
        default_source="pubmed-rct",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )

@register_benchmark()
def load_squad(force_rebuild: bool = False) -> list[EvalData]:
    """Load PubMed-RCT subset from MemAlpha into EvalData format."""
    return _load_parquet_subset(
        parquet_path=MEMALPHA_PARQUET_PATH,
        cache_path="data/processed_squad.json",
        data_source_value="squad",
        task_prefix="squad",
        default_source="squad",
        id_keys=["instance_id"],
        force_rebuild=force_rebuild,
    )

@register_benchmark()
def load_msc(force_rebuild=False, batch_size=5):
    """
    Load MSC-Self-Instruct dataset and optionally merge samples into larger batches.
    
    Structure per base sample:
    - 500 samples from Session 5 conversations
    - Each sample has 5 sessions total (previous 4 in previous_dialogs + current session 5 in dialog)
    - Chunks: Each session as one chunk (5 chunks total)
    - Questions: self_instruct QA pairs asked at position 4 (after all 5 sessions)
    
    Args:
        force_rebuild: skip cached JSON
        batch_size: number of base samples to merge into one EvalData record
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    suffix = "" if batch_size == 1 else f"_batch{batch_size}"
    file_path = f"data/processed_msc{suffix}.json"
    if os.path.exists(file_path) and not force_rebuild:
        return load_from_path(file_path)
    
    from datasets import load_dataset
    raw = load_dataset("MemGPT/MSC-Self-Instruct", split="train")
    
    base_samples = []
    
    for idx, item in enumerate(raw):
        task_id = f"msc_{item['metadata']['initial_data_id']}"
        
        # Build chunks from all sessions
        chunks = []
        
        # Process previous dialogs (Sessions 1-4)
        for prev_dialog in item['previous_dialogs']:
            session_conv = ""
            for turn in prev_dialog['dialog']:
                # Determine speaker (alternating, starting with Speaker 1)
                turn_idx = prev_dialog['dialog'].index(turn)
                speaker = "Speaker 1" if turn_idx % 2 == 0 else "Speaker 2"
                session_conv += f'{speaker} said, "{turn["text"]}"\n'
            
            chunks.append(session_conv.strip())
        
        # Process current dialog (Session 5)
        session_conv = ""
        for turn in item['dialog']:
            speaker = turn['id']  # Already has 'Speaker 1' or 'Speaker 2'
            session_conv += f'{speaker} said, "{turn["text"]}"\n'
        
        chunks.append(session_conv.strip())
        
        # Create question from self_instruct
        question = Question(
            qid=f"msc_{item['metadata']['initial_data_id']}_{idx}",
            query=item['self_instruct']['B'],
            answer=item['self_instruct']['A'],
            position=4,  # Ask after all 5 sessions (0-indexed, so position 4 = after chunk 4)
            category='memory-recall'
        )
        
        base_samples.append(EvalData(
            task_id=task_id,
            questions=[question],
            chunks=chunks
        ))

    if batch_size == 1:
        processed = base_samples
    else:
        processed = []
        for batch_index, start in enumerate(range(0, len(base_samples), batch_size)):
            batch = base_samples[start:start + batch_size]
            combined_chunks = []
            combined_questions = []
            chunk_offset = 0

            for sample in batch:
                combined_chunks.extend(sample.chunks)
                for q in sample.questions:
                    combined_questions.append(Question(
                        qid=q.qid,
                        query=q.query,
                        answer=q.answer,
                        position=q.position + chunk_offset,
                        category=q.category
                    ))
                chunk_offset += len(sample.chunks)

            batch_task_id = f"msc_batch_{batch_index}_{batch[0].task_id}_to_{batch[-1].task_id}"
            processed.append(EvalData(
                task_id=batch_task_id,
                questions=combined_questions,
                chunks=combined_chunks
            ))

    # Save processed data
    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed

def load_from_path(path: str) -> list[EvalData]:
    """Load EvalData from a given JSON file path."""
    if os.environ.get("USE_SMALL_VALSETS") == "1":
        path = path.replace("/", "/small_valsets/")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    processed_dataset = json.load(open(path))
    return [EvalData(**data) for data in processed_dataset]

@register_benchmark()
def load_synth(suf, force_rebuild=False) -> list[EvalData]:
    if os.path.exists(f"data/processed_synth-{suf}.json") and not force_rebuild:
        return load_from_path(f"data/processed_synth-{suf}.json")
    from synthv1_async import main
    num_questions = 7000 if "train" in suf else 100
    main(num_sessions=int(''.join(filter(str.isdigit, suf))), out=f"data/processed_synth-{suf}.json", no_diversify=True, num_questions=num_questions)
    return load_from_path(f"data/processed_synth-{suf}.json")

@register_benchmark()
def load_convomem(force_rebuild=False) -> list[EvalData]:
    """Load ConvoMem benchmark data from pre_mixed_testcases.

    Filters test cases with total conversation length > 2,180,000 characters (~10k questions).
    Each test case becomes one EvalData with multiple questions from evidenceItems.
    Each conversation becomes one chunk.
    """
    file_path = "data/processed_convomem.json"
    if os.path.exists(file_path) and not force_rebuild:
        convomem_data = load_from_path(file_path)
        for item in convomem_data:
            item.questions = [q for q in item.questions if q.category == "user_evidence"]
        convomem_data = [item for item in convomem_data if item.questions]
        return convomem_data

    from pathlib import Path
    base_dir = Path("./data/raw/ConvoMem/core_benchmark/pre_mixed_testcases")
    json_files = sorted(p.as_posix() for p in base_dir.rglob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    processed = []
    total_items = 0
    filtered_items = 0

    for json_path in json_files:
        # Extract category from directory path
        # e.g., .../assistant_facts_evidence/6_evidence/batched_001.json -> assistant_facts_evidence
        path_parts = Path(json_path).parts
        try:
            category_idx = path_parts.index("pre_mixed_testcases") + 1
            category = path_parts[category_idx] if category_idx < len(path_parts) else "unknown"
        except (ValueError, IndexError):
            category = "unknown"

        total_data = json.load(open(json_path))

        for item_idx, item in enumerate(total_data):
            total_items += 1

            # Calculate total character length using the same method as reference code
            context = '\n'.join([mess['text'] for conv in item['conversations'] for mess in conv['messages']])
            if len(context) <= 2180000:
                continue

            filtered_items += 1

            # Build chunks from conversations (each conversation becomes one chunk)
            chunks = []
            for conv in item['conversations']:
                conv_text = ''
                for msg in conv['messages']:
                    speaker = msg.get('speaker', 'unknown')
                    text = msg.get('text', '')
                    conv_text += f'{speaker}: {text}\n'
                chunks.append(conv_text.strip())

            # Build questions from evidenceItems
            questions = []
            final_position = len(chunks) - 1

            # Generate unique task_id
            file_name = Path(json_path).stem
            task_id = f"convomem_{category}_{file_name}_{item_idx}"

            for q_idx, evidence_item in enumerate(item['evidenceItems']):
                question_text = evidence_item.get('question', '')
                answer_text = evidence_item.get('answer', '')

                if not question_text or category != "user_evidence":
                    continue

                questions.append(Question(
                    qid=f"{task_id}_{q_idx}",
                    query=question_text,
                    answer=answer_text,
                    position=final_position,
                    category=category
                ))

            if questions:  # Only add if there are questions
                evaldata = EvalData(
                    task_id=task_id,
                    questions=questions,
                    chunks=chunks
                )
                processed.append(evaldata)

    print(f"Total items processed: {total_items}")
    print(f"Items with length > 1M: {filtered_items}")
    print(f"Final EvalData entries: {len(processed)}")

    # Save processed data
    json.dump([item.model_dump() for item in processed], open(file_path, "w"), indent=4, ensure_ascii=False)
    return processed

AVAILABLE_BENCHMARKS = sorted(BENCHMARK_REGISTRY)
print(f"[EvalDataset] Registered benchmarks: {', '.join(AVAILABLE_BENCHMARKS)}")
# banking77, booksum, clinic, hotpotqa, locomo, longmemeval, memalpha, msc, nlu, perltqa, pubmed_rct, trec_coarse, trec_fine

if __name__ == '__main__':
    # load_synth("ss2")
    # load_synth("ss5")
    # load_synth("ss10")
    # load_synth("ss10_train")
    load_longmemeval()