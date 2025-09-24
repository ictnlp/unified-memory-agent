import json
import os
from pydantic import BaseModel
import random
CONV_START_PROMPT = "Below is a conversation between {} and {}.\n\n"

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

def load_locomo(force_rebuild=False):
    # 有五类QA
    # 1. multi hop
    # 2. temporal
    # 3. open domain
    # 4. single hop
    # 5. adversarial
    file_path = "data/processed_locomo.json"
    if os.path.exists(file_path) and not force_rebuild:
        processed_dataset = json.load(open(file_path))
        return [EvalData(**data) for data in processed_dataset]
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
            sess_position = random.randint(last_sess_id, num_sessions)
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

def load_longmemeval(force_rebuild=False, oracle=False):
    file_path = "data/processed_longmemeval_oracle.json" if oracle else "data/processed_longmemeval.json"
    if os.path.exists(file_path) and not force_rebuild:
        processed_dataset = json.load(open(file_path))
        return [EvalData(**data) for data in processed_dataset]
    
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

def load_hotpotqa(force_rebuild=False, num_docs: int=200, num_queries: int=5, num_samples: int=100):
    file_path = f"data/processed_hotpotqa_{num_docs}_{num_queries}_{num_samples}.json"
    if os.path.exists(file_path) and not force_rebuild:
        processed_dataset = json.load(open(file_path))
        return [EvalData(**data) for data in processed_dataset]
    
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


if __name__ == '__main__':
    # locomo = load_locomo(True)
    # longmemeval = load_longmemeval(True)
    hotpotqa = load_hotpotqa(True, num_docs=10, num_queries=3, num_samples=5)  # Small test
    print(f"Loaded {len(hotpotqa)} HotpotQA samples")
    if hotpotqa:
        print(f"Sample: {hotpotqa[0].task_id}, Questions: {len(hotpotqa[0].questions)}, Chunks: {len(hotpotqa[0].chunks)}")