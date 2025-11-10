import json
import numpy as np

# Load locomo dataset
locomo = json.load(open("/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/processed_locomo.json"))

# Take only the first sample
first_sample = locomo[0]
print(f"Processing sample: {first_sample['task_id']}")
print(f"Total questions: {len(first_sample['questions'])}")
print(f"Total chunks: {len(first_sample['chunks'])}")

# Prepare data structure
data = {
    "data_source": [],
    "prompt": [],
    "context": [],
    "reward_model": [],
    "extra_info": [],
    "agent_name": []
}

# Group questions into batches of 10
questions = first_sample['questions']
chunks = first_sample['chunks']
batch_size = 10

for i in range(0, len(questions), batch_size):
    batch_questions = questions[i:i+batch_size]
    
    # Extract question texts and answers
    question_texts = [q['query'] for q in batch_questions]
    answers = [q['answer'] for q in batch_questions]
    
    # Build prompt field: single dict with list of questions
    prompt_item = {
        'content': question_texts,
        'role': 'user'
    }
    
    # Build context: join all chunks, remove duplicate headers
    # Keep "Below is a conversation..." only in the first chunk
    context_parts = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            # Keep the first chunk as is
            context_parts.append(chunk)
        else:
            # Remove the header line if it exists
            if chunk.startswith('Below is a conversation between'):
                # Find the end of the first line
                first_newline = chunk.find('\n')
                if first_newline != -1:
                    # Skip the header line
                    context_parts.append(chunk[first_newline+1:])
                else:
                    context_parts.append(chunk)
            else:
                context_parts.append(chunk)
    
    context_str = '\n\n'.join(context_parts)
    
    # Build reward_model field
    reward_model = {
        'ground_truth': answers
    }
    
    # Build extra_info field
    filename = "/mnt/pfs-guan-ssai/nlu/zhangkehao/verl/memagent/store/memory_store.jsonl"
    
    # Convert chunks to numpy array for consistency with reference data
    chunks_array = np.array(chunks, dtype=object)
    
    extra_info = {
        'index': i // batch_size,
        'num_chunks': len(chunks),
        'question': question_texts,  # Store questions for reference
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
    data['data_source'].append('locomo')
    data['prompt'].append(np.array([prompt_item], dtype=object))  # Wrap in array for consistency
    data['context'].append(context_str)
    data['reward_model'].append(reward_model)
    data['extra_info'].append(extra_info)
    data['agent_name'].append('tool_mem_agent')

print(f"\nGenerated {len(data['data_source'])} training samples")
print(f"Each sample has {batch_size} questions (last batch may have fewer)")

# Save to parquet file
import pandas as pd
df = pd.DataFrame(data)
output_file = "/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/locomo_train_verl.parquet"
df.to_parquet(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Print first sample for verification
print("\n=== First Sample ===")
print(f"data_source: {df.iloc[0]['data_source']}")
print(f"prompt: {df.iloc[0]['prompt']}")
print(f"context length: {len(df.iloc[0]['context'])} chars")
print(f"reward_model: {df.iloc[0]['reward_model']}")
print(f"agent_name: {df.iloc[0]['agent_name']}")
print(f"extra_info keys: {df.iloc[0]['extra_info'].keys()}")
