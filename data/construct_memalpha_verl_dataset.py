import json
import numpy as np
import pandas as pd
from datasets import load_dataset

# Load Memalpha dataset
print("Loading Memalpha dataset...")
ds = load_dataset("YuWangX/Memalpha", split="train")
print(f"Loaded {len(ds)} samples from Memalpha")

# Filter out lme_train samples
print("Filtering out lme_train samples...")
ds = ds.filter(lambda x: x['data_source'] != 'lme_train')
print(f"After filtering: {len(ds)} samples remaining")

# Prepare data structure
data = {
    "data_source": [],
    "prompt": [],
    "context": [],
    "reward_model": [],
    "extra_info": [],
    "agent_name": []
}

total_generated = 0

for sample_idx, sample in enumerate(ds):
    # Parse JSON fields
    chunks = json.loads(sample['chunks'])
    questions_and_answers = json.loads(sample['questions_and_answers'])
    
    # Extract questions and answers
    questions = [qa['question'] for qa in questions_and_answers]
    answers = [qa['answer'] for qa in questions_and_answers]
    
    num_questions = len(questions)
    original_data_source = sample['data_source']
    
    print(f"\nProcessing sample {sample_idx}: {original_data_source}, {num_questions} questions, {len(chunks)} chunks")
    
    # Determine batch size and grouping
    if num_questions > 10:
        # Group into batches of 10
        batch_size = 10
    else:
        # All questions in one batch
        batch_size = num_questions
    
    # Build context: prepend prompt, then join all chunks
    context_str = sample['prompt'] + '\n\n' + '\n\n'.join(chunks)
    
    # Convert chunks to numpy array
    chunks_array = np.array(chunks, dtype=object)
    
    # Group questions into batches
    for i in range(0, num_questions, batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]
        
        # Build prompt field: single dict with list of questions
        prompt_item = {
            'content': batch_questions,
            'role': 'user'
        }
        
        # Build reward_model field
        reward_model = {
            'ground_truth': batch_answers
        }
        
        # Build extra_info field
        filename = "/mnt/pfs-guan-ssai/nlu/zhangkehao/verl/memagent/store/memory_store.jsonl"
        
        extra_info = {
            'index': total_generated,
            'num_chunks': len(chunks),
            'question': batch_questions,  # Store questions for reference
            'original_sample_id': sample['instance_id'],
            'tools_kwargs': {
                'memory_add': {
                    'create_kwargs': {
                        'filename': filename
                    }
                },
                'memory_bm25_retrieve': {
                    'create_kwargs': {
                        'chunks': chunks_array
                    }
                },
                'memory_delete': {
                    'create_kwargs': {
                        'filename': filename
                    }
                },
                'memory_embedding_retrieve': {
                    'create_kwargs': {
                        'chunks': chunks_array
                    }
                },
                'memory_key_retrieve': {
                    'create_kwargs': {
                        'filename': filename
                    }
                },
                'memory_list': {
                    'create_kwargs': {
                        'filename': filename
                    }
                },
                'memory_update': {
                    'create_kwargs': {
                        'filename': filename
                    }
                }
            }
        }
        
        # Append to data
        data['data_source'].append(f'memalpha_{original_data_source}')
        data['prompt'].append(np.array([prompt_item], dtype=object))  # Wrap in array
        data['context'].append(context_str)
        data['reward_model'].append(reward_model)
        data['extra_info'].append(extra_info)
        data['agent_name'].append('tool_mem_agent')
        
        total_generated += 1

print(f"\n{'='*60}")
print(f"Total training samples generated: {total_generated}")
print(f"From {len(ds)} original Memalpha samples")

# Save to parquet file
df = pd.DataFrame(data)
output_file = "/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/memalpha_train_verl.parquet"
df.to_parquet(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Print statistics
print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Unique data sources: {df['data_source'].nunique()}")

print("\n=== Sample data sources ===")
print(df['data_source'].value_counts().head(10))

print("\n=== Questions per sample ===")
questions_per_sample = [len(df.iloc[i]['prompt'][0]['content']) for i in range(min(10, len(df)))]
print(f"First 10 samples: {questions_per_sample}")

print("\n=== First Sample Preview ===")
print(f"data_source: {df.iloc[0]['data_source']}")
print(f"num_questions: {len(df.iloc[0]['prompt'][0]['content'])}")
print(f"num_answers: {len(df.iloc[0]['reward_model']['ground_truth'])}")
print(f"context length: {len(df.iloc[0]['context'])} chars")
print(f"num_chunks: {df.iloc[0]['extra_info']['num_chunks']}")
print(f"agent_name: {df.iloc[0]['agent_name']}")
