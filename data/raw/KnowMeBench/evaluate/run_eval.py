import os
import json
import re
from typing import List, Dict, Any
from openai import AsyncOpenAI
import asyncio

# ================= Configuration =================
# Default Judge Model as specified in Section 4.4 of the paper (GPT-4o)
DEFAULT_JUDGE_MODEL = "gpt-4o"

# Path to the prompt template file
PROMPT_FILE = "evaluate prompt.md"
# =================================================

# Global client - will be set by run() function
client = None

def parse_prompt_file(file_path: str) -> Dict[str, str]:
    """
    Parses the Markdown file to extract prompt templates for different task types.
    
    Structure expectation:
    # type Task Name
    [Prompt Content]
    
    Args:
        file_path: Path to the evaluate prompt.md file.
        
    Returns:
        Dictionary mapping task types to their specific prompt templates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content by the '# type' header
    sections = re.split(r'^# type\s+(.+)$', content, flags=re.MULTILINE)
    
    prompts = {}
    # re.split results in [header, type_name, content, type_name, content...]
    # We iterate starting from index 1 with a step of 2
    for i in range(1, len(sections), 2):
        task_types_raw = sections[i].strip()
        prompt_content = sections[i+1].strip()
        
        # Handle cases where multiple tasks share one prompt (separated by comma or Chinese comma)
        # e.g., "Mnestic Trigger Analysis、Temporal Reasoning"
        task_types = [t.strip() for t in re.split(r'[、,]', task_types_raw)]
        
        for t in task_types:
            prompts[t] = prompt_content

    print(f"✅ Successfully loaded {len(prompts)} prompt templates from {file_path}")
    return prompts

async def evaluate_single_item(item: Dict, prompt_template: str, model: str) -> Dict:
    """
    Evaluates a single model response using the LLM-as-a-Judge protocol (async version).

    Args:
        item: Dictionary containing 'question', 'reference_answer', and 'model_answer'.
        prompt_template: The specific prompt string for this task type.
        model: The judge model to use (e.g., gpt-4o).

    Returns:
        Dictionary containing the evaluation score and reasoning.
    """
    # 1. Fill the template placeholders
    user_question = item.get('question', '')
    ref_answer = item.get('reference_answer', '')
    model_ans = item.get('model_answer', '')

    # Replace placeholders defined in the prompt file
    final_prompt = prompt_template.replace('{{question}}', str(user_question))\
                                  .replace('{{reference_answer}}', str(ref_answer))\
                                  .replace('{{model_answer}}', str(model_ans))

    # 2. Call the LLM Judge (async)
    # As per paper Section 4.4, we use a rigorous rubric-based scoring.
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an impartial judge evaluating AI model outputs based on strict criteria."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0, # Deterministic output for reproducibility
            response_format={"type": "json_object"} # Enforce JSON format for parsing
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        return {
            "id": item.get('id'),
            "task_type": item.get('task_type'),
            "score": result_json.get('score'),
            "reasoning": result_json.get('reasoning'),
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Error processing Item ID {item.get('id')}: {e}")
        return {
            "id": item.get('id'),
            "task_type": item.get('task_type'),
            "score": 0,
            "reasoning": f"Evaluation Error: {str(e)}",
            "status": "error"
        }

def run(input_data: List[Dict], judge_model: str = DEFAULT_JUDGE_MODEL, scoring_client: AsyncOpenAI = None) -> Dict:
    """
    KnowMe-Bench LLM-as-a-Judge Evaluator

    Args:
        input_data: List of dicts containing [question, reference_answer, model_answer, task_type].
        judge_model: The OpenAI model to serve as the judge (default: gpt-4o).
        scoring_client: AsyncOpenAI client to use for scoring.

    Returns:
        Dict with evaluation results.
    """
    global client
    client = scoring_client

    if client is None:
        raise ValueError("scoring_client must be provided")

    # Load prompt templates
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(script_dir, PROMPT_FILE)
    prompts_map = parse_prompt_file(prompt_file_path)

    # Create evaluation tasks
    async def evaluate_all():
        tasks = []
        for item in input_data:
            task_type = item.get('task_type')
            if task_type not in prompts_map:
                continue
            tasks.append(evaluate_single_item(item, prompts_map[task_type], judge_model))
        return await asyncio.gather(*tasks)

    results = asyncio.run(evaluate_all())

    # Calculate stats
    valid_scores = [r['score'] for r in results if r.get('status') == 'success' and isinstance(r.get('score'), (int, float))]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    return {
        "meta": {
            "judge_model": judge_model,
            "total_items": len(input_data),
            "evaluated_items": len(valid_scores),
            "average_score": round(avg_score, 4)
        },
        "details": results
    }

