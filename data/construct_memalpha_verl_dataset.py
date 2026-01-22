import json
import numpy as np
import pandas as pd
import random
from datasets import load_dataset
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--val', action='store_true', help='If set, process validation set instead of training set')
args = argparser.parse_args()

# Load Memalpha dataset
print("Loading Memalpha dataset...")
ds = load_dataset("YuWangX/Memalpha-full", split="validation" if args.val else "train")
print(f"Loaded {len(ds)} samples from Memalpha")

# Filter out lme_train and hotpotqa samples
print("Filtering out lme_train and hotpotqa samples...")
ds = ds.filter(lambda x: x['data_source'] != 'lme_train' and x['data_source'] != 'hotpotqa')
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
    
    # Randomly sample up to 10 questions
    if num_questions > 10:
        # Randomly sample 10 questions
        sample_indices = random.sample(range(num_questions), 10)
        sample_indices.sort()  # Sort to maintain some order
        sampled_questions = [questions[i] for i in sample_indices]
        sampled_answers = [answers[i] for i in sample_indices]
        print(f"  Sampled 10 questions from {num_questions} (indices: {sample_indices})")
    else:
        # Use all questions
        sampled_questions = questions
        sampled_answers = answers
    
    # Build context: prepend prompt, then join all chunks
    context_str = sample['prompt'] + '\n\n' + '\n\n'.join(chunks)
    
    # Convert chunks to numpy array
    chunks_array = np.array(chunks, dtype=object)
    
    # Build prompt field: single dict with list of questions
    prompt_item = {
        'content': sampled_questions,
        'role': 'user'
    }
    
    # Build reward_model field
    reward_model = {
        'ground_truth': sampled_answers
    }
    
    # Build extra_info field
    filename = "./tmp/verl_train/memory_store.jsonl"
    
    extra_info = {
        'index': total_generated,
        'num_chunks': len(chunks),
        'question': sampled_questions,  # Store questions for reference
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
output_file = f"./train/memalphafull-train/{'validation' if args.val else 'train'}.parquet"
if args.val:
    df = (
        df.groupby("data_source", group_keys=False)
        .apply(lambda group: group.sample(n=20, random_state=42) if len(group) >= 20 else group)
        .reset_index(drop=True)
    )
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
