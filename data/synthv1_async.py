import asyncio
import os
import json
import uuid
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    import fire
except ImportError:  # pragma: no cover - optional dependency for CLI usage
    fire = None

# Consumption scene definitions (scene name: subscene list)
CONSUMPTION_SCENES = {
    "Dining": ["Fast Food", "Restaurant", "Coffee", "Bubble Tea", "BBQ", "Hot Pot", "Snacks", "Takeout"],
    "Transportation": ["Subway", "Bus", "Taxi", "Gas", "Parking", "Train", "Flight"],
    "Shopping": ["Clothing", "Electronics", "Daily Necessities", "Cosmetics", "Books", "Groceries", "Furniture"],
    "Entertainment": ["Movie", "KTV", "Gaming", "Gym", "Travel", "Concert", "Escape Room"],
    "Utilities": ["Water & Electricity", "Property Fee", "Phone Bill", "Internet", "Gas Bill", "Rent"],
    "Medical": ["Medicine", "Doctor Visit", "Health Checkup", "Dental", "Glasses"],
    "Education": ["Training Course", "Books & Materials", "Online Course", "Exam Registration", "Tuition"],
    "Other": ["Transfer", "Red Envelope", "Donation", "Pet", "Beauty & Salon"]
}

# DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# API_CONFIG = {
#     "base_url": "http://127.0.0.1:8000/v1",
#     "api_key": "EMPTY",
#     "max_retries": 100
# }

DEFAULT_MODEL = "gemini-3-pro-preview"
API_CONFIG = {
    "base_url": "http://api-hub.inner.chj.cloud/llm-gateway/v1",
    "api_key": "sk-EMPTY",
    "default_headers": {
        "BCS-APIHub-RequestId": str(uuid.uuid4()),
        "X-CHJ-GWToken": 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI5OHFnRkVlcEFVeEM4Qk91UUZJVDNXQUwwTG9OV0NQaSJ9.vTX_oU2rXrrgkPK5_RU76b2gRLpwlCxZ346mvCdug7A',
    },
    "max_retries": 100
}

@dataclass
class Transaction:
    """Transaction record"""
    date: str  # YYYY-MM-DD
    scene: str  # Consumption scene
    subscene: str  # Subscene
    amount: float  # Amount
    description: str  # Description
    session_index: int  # Which session it belongs to

    def to_dict(self):
        return {
            "date": self.date,
            "scene": self.scene,
            "subscene": self.subscene,
            "amount": self.amount,
            "description": self.description,
            "session_index": self.session_index
        }


def get_client():
    return AsyncOpenAI(**API_CONFIG)


_CALL_LLM_SEMAPHORE: Optional[asyncio.Semaphore] = None


async def call_llm(client, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """Call LLM asynchronously, offloading sync clients if needed."""
    async def _invoke():
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=8192,
        )
        return resp.choices[0].message.content

    semaphore = _CALL_LLM_SEMAPHORE
    if semaphore is None:
        return await _invoke()

    async with semaphore:
        return await _invoke()


def parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON response"""
    try:
        return json.loads(content)
    except Exception:
        if repair_json:
            return json.loads(repair_json(content))
        raise ValueError(f"LLM response is not valid JSON: {content}")


def generate_dates(start_date: datetime, end_date: datetime, count: int) -> List[str]:
    """Generate random dates within the time range"""
    dates = []
    delta = (end_date - start_date).days
    
    # Generate unique random dates
    random_days = random.sample(range(delta + 1), count)
    for day_offset in sorted(random_days):
        date = start_date + timedelta(days=day_offset)
        dates.append(date.strftime("%Y-%m-%d"))
    
    return dates


async def generate_session(client, model: str, date: str, session_index: int, 
                           is_first_session: bool = False,
                           min_turns: int = 20, max_turns: int = 50,
                           min_scenes: int = 1, max_scenes: int = 10) -> Tuple[List[Dict[str, str]], List[Transaction]]:
    """Generate dialogue and transaction records for one session asynchronously."""
    
    # Randomly select consumption scenes
    num_scenes = random.randint(min_scenes, max_scenes)
    selected_scenes = []
    
    all_scene_pairs = [(scene, subscene) for scene, subscenes in CONSUMPTION_SCENES.items() 
                       for subscene in subscenes]
    selected_scene_pairs = random.sample(all_scene_pairs, min(num_scenes, len(all_scene_pairs)))
    
    for scene, subscene in selected_scene_pairs:
        selected_scenes.append({"scene": scene, "subscene": subscene})
    
    # Different prompt for first session vs. subsequent sessions
    if is_first_session:
        context_instruction = """This is the FIRST conversation session. The user should:
- Start with a greeting and introduce their intention to track expenses
- Then proceed to record the consumption items"""
    else:
        context_instruction = """This is a CONTINUATION of an ongoing expense tracking conversation. The user should:
- Directly start recording new expenses without greeting or re-introducing themselves
- Act as if they are continuing from previous sessions
- Use natural transitions like "I have a few more expenses to log" or directly mention the items"""
    
    # Generate dialogue prompt
    prompt = f"""Please generate a natural conversation between a user and an AI assistant about expense tracking.

Context:
- Date of conversation: {date}
- {context_instruction}
- The user wants to record the following consumption scenes: {json.dumps(selected_scenes, ensure_ascii=False)}
- The conversation should be natural and fluid, containing {min_turns}-{max_turns} turns
- Include confirmations, clarifications, and other contextual content as needed
- Each consumption scene needs a specific amount (you decide a reasonable amount)

Please return in JSON format with two fields:
1. "dialogue": List of conversations, each item contains {{"role": "user"/"assistant", "content": "dialogue content"}}
2. "transactions": List of transaction records, each item contains {{"scene": "scene", "subscene": "subscene", "amount": amount(number), "description": "brief description", "date": "YYYY-MM-DD"}}

Notes:
- The dialogue should be natural, not too mechanical
- Amount must be a number (e.g., 45.5), without currency symbols
- Each item in transactions must be reflected in the dialogue
- For non-first sessions, avoid greetings like "Hi there!" or "Can you help me track expenses?"
- IMPORTANT: When mentioning expenses in the dialogue, always explicitly state the scene category (e.g., "Dining", "Transportation", "Shopping") rather than just the subscene (e.g., don't just say "Coffee", say "Dining - Coffee" or "spent on dining for coffee")

Please return JSON only, no other content."""

    messages = [
        {"role": "system", "content": "You are an AI assistant that helps generate conversational data."},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            response = await call_llm(client, model, messages, temperature=0.8)
            parsed = parse_json_response(response)
            # Build Transaction objects
            transactions = []
            for tx in parsed.get("transactions", []):
                transactions.append(Transaction(
                    date=tx["date"],
                    scene=tx["scene"],
                    subscene=tx["subscene"],
                    amount=float(tx["amount"]),
                    description=tx["description"],
                    session_index=session_index
                ))
            
            # Build dialogue turns
            turns = []
            for turn in parsed.get("dialogue", []):
                turns.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
            break
        except Exception as e:
            print(f"Error generating session for date {date}: {e}, retrying...")
            await asyncio.sleep(1)
            continue
    
    return turns, transactions


async def generate_questions(client, model: str, transactions: List[Transaction], dates: List[str],
                             diversify: bool = True, position: int = None) -> List[Dict[str, Any]]:
    """
    Generate questions asynchronously.
    Fixed generation: Type 1-7 each generates 5 questions, Type 8 generates 1 question.
    Total: up to 36 questions per call (may be less if data doesn't meet requirements).

    Requirements for full 36 questions:
    - len(dates) >= 2 (for Type 1-6)
    - scene_stats exists (for Type 4)
    - transactions exist (for Type 6)
    - date_count exists for at least one scene (for Type 5)

    Returns actual number of questions generated (may be < 36).
    """

    questions = []
    task_id = uuid.uuid4().hex[:8]

    # Statistics by scene
    scene_stats = defaultdict(float)
    subscene_stats = defaultdict(float)
    date_stats = defaultdict(float)
    date_scene_stats = defaultdict(lambda: defaultdict(float))

    for tx in transactions:
        scene_stats[tx.scene] += tx.amount
        subscene_stats[f"{tx.scene}-{tx.subscene}"] += tx.amount
        date_stats[tx.date] += tx.amount
        date_scene_stats[tx.date][tx.scene] += tx.amount

    total_amount = sum(tx.amount for tx in transactions)

    # Question templates
    question_specs = []

    # Fixed: each of Type 1-7 generates 5 questions, Type 8 generates 1 question
    q_per_type = 5

    # Type 1: Total spending in a specific scene within a time range
    for _ in range(q_per_type):
        if len(dates) >= 2:
            start_idx = random.randint(0, len(dates) - 2)
            end_idx = random.randint(start_idx + 1, len(dates) - 1)
            start_date = dates[start_idx]
            end_date = dates[end_idx]
            
            scene = random.choice(list(CONSUMPTION_SCENES.keys()))
            
            # Calculate answer
            amount = sum(tx.amount for tx in transactions 
                        if tx.scene == scene and start_date <= tx.date <= end_date)
            
            question_specs.append({
                "template": f"How much was spent on {scene} from {start_date} to {end_date}?",
                "answer": f"{amount:.2f}",
                "category": "time_range_scene_amount"
            })
    
    # Type 2: Total spending in multiple scenes within a time range
    for _ in range(q_per_type):
        if len(dates) >= 2:
            start_idx = random.randint(0, len(dates) - 2)
            end_idx = random.randint(start_idx + 1, len(dates) - 1)
            start_date = dates[start_idx]
            end_date = dates[end_idx]
            
            num_scenes = random.randint(2, 3)
            scenes = random.sample(list(CONSUMPTION_SCENES.keys()), num_scenes)
            
            amount = sum(tx.amount for tx in transactions 
                        if tx.scene in scenes and start_date <= tx.date <= end_date)
            
            scenes_str = ", ".join(scenes)
            question_specs.append({
                "template": f"How much was spent on {scenes_str} in total from {start_date} to {end_date}?",
                "answer": f"{amount:.2f}",
                "category": "time_range_multiple_scenes_amount"
            })
    
    # Type 3: Total spending within a time range
    for _ in range(q_per_type):
        if len(dates) >= 2:
            start_idx = random.randint(0, len(dates) - 2)
            end_idx = random.randint(start_idx + 1, len(dates) - 1)
            start_date = dates[start_idx]
            end_date = dates[end_idx]
            
            amount = sum(tx.amount for tx in transactions 
                        if start_date <= tx.date <= end_date)
            
            question_specs.append({
                "template": f"What was the total spending from {start_date} to {end_date}?",
                "answer": f"{amount:.2f}",
                "category": "time_range_total_amount"
            })
    
    # Type 4: Which scene had the most spending
    for _ in range(q_per_type):
        if len(dates) >= 2 and scene_stats:
            start_idx = random.randint(0, len(dates) - 2)
            end_idx = random.randint(start_idx + 1, len(dates) - 1)
            start_date = dates[start_idx]
            end_date = dates[end_idx]

            range_scene_stats = defaultdict(float)
            for tx in transactions:
                if start_date <= tx.date <= end_date:
                    range_scene_stats[tx.scene] += tx.amount

            if range_scene_stats:
                # 找出所有最大值的场景（可能有多个）
                max_amount = max(range_scene_stats.values())
                max_scenes = [scene for scene, amount in range_scene_stats.items() if amount == max_amount]

                # 答案是所有最大值场景，按字母排序以保证一致性
                answer = ", ".join(sorted(max_scenes)) if len(max_scenes) > 1 else max_scenes[0]

                # 列出所有可能的scene类别以明确问题
                all_scenes = ", ".join(list(CONSUMPTION_SCENES.keys()))

                question_specs.append({
                    "template": f"Which main consumption scene category (such as {all_scenes}) had the highest spending from {start_date} to {end_date}? Please answer with the main category name only, not the subcategory. If there are multiple answers, output all of them separated by commas.",
                    "answer": answer,
                    "category": "max_scene"
                })

    # Type 5: Date with the most transactions in a specific scene
    for _ in range(q_per_type):
        scene = random.choice(list(CONSUMPTION_SCENES.keys()))
        date_count = defaultdict(int)

        for tx in transactions:
            if tx.scene == scene:
                date_count[tx.date] += 1

        if date_count:
            # 找出所有交易次数最多的日期（可能有多个）
            max_count = max(date_count.values())
            max_dates = [date for date, count in date_count.items() if count == max_count]

            # 答案是所有最大值日期，按时间排序
            answer = ", ".join(sorted(max_dates)) if len(max_dates) > 1 else max_dates[0]

            question_specs.append({
                "template": f"On which date were there the most {scene} transactions? If there are multiple answers, output all of them separated by commas.",
                "answer": answer,
                "category": "max_frequency_date"
            })
    
    # Type 6: Largest single transaction
    for _ in range(q_per_type):
        if len(dates) >= 2 and transactions:
            start_idx = random.randint(0, len(dates) - 2)
            end_idx = random.randint(start_idx + 1, len(dates) - 1)
            start_date = dates[start_idx]
            end_date = dates[end_idx]
            
            range_txs = [tx for tx in transactions if start_date <= tx.date <= end_date]
            
            if range_txs:
                max_tx = max(range_txs, key=lambda x: x.amount)
                question_specs.append({
                    "template": f"What was the largest single transaction amount from {start_date} to {end_date}?",
                    "answer": f"{max_tx.amount:.2f}",
                    "category": "max_single_amount"
                })
    
    # Type 7: Spending on a specific scene on a specific date
    for _ in range(q_per_type):
        date = random.choice(dates)
        scene = random.choice(list(CONSUMPTION_SCENES.keys()))
        
        amount = sum(tx.amount for tx in transactions 
                    if tx.date == date and tx.scene == scene)
        
        question_specs.append({
            "template": f"How much was spent on {scene} on {date}?",
            "answer": f"{amount:.2f}",
            "category": "single_date_scene_amount"
        })
    
    # Type 8: Total spending across all records (only once)
    question_specs.append({
        "template": f"What is the total spending across all records?",
        "answer": f"{total_amount:.2f}",
        "category": "total_amount"
    })

    # Remove duplicates before diversification step (take first per template)
    spec_by_template: Dict[str, Dict[str, Any]] = {}
    for spec in question_specs:
        spec_by_template.setdefault(spec["template"], spec)
    question_specs = list(spec_by_template.values())

    # Use LLM to diversify questions
    if diversify:
        print("Diversifying questions with LLM...")
        
        batch_size = 10

        async def diversify_batch(batch_index: int) -> None:
            start = batch_index * batch_size
            batch = question_specs[start:start + batch_size]
            if not batch:
                return

            questions_text = "\n".join([f"{idx + 1}. {item['template']}" for idx, item in enumerate(batch)])

            prompt = f"""Please rewrite the following questions to make them more diverse and natural, while keeping the core meaning unchanged:

{questions_text}

Requirements:
1. Keep key information such as dates and scene names unchanged
2. Use different phrasings and expressions
3. Make the questions more conversational and natural
4. Do not change what the question is asking

Please return in JSON format: {{"questions": ["question 1", "question 2", ...]}}

Please return JSON only, no other content."""

            messages = [
                {"role": "system", "content": "You are a question diversification expert."},
                {"role": "user", "content": prompt},
            ]

            try:
                response = await call_llm(client, model, messages, temperature=0.8)
                parsed = parse_json_response(response)
                diversified = parsed.get("questions", [])

                for offset, new_q in enumerate(diversified):
                    target_index = start + offset
                    if target_index < len(question_specs):
                        question_specs[target_index]["query"] = new_q
            except Exception as exc:
                print(f"Diversification failed for batch {batch_index}: {exc}, using original questions")
                for offset, item in enumerate(batch):
                    question_specs[start + offset]["query"] = item["template"]

        total_batches = (len(question_specs) + batch_size - 1) // batch_size
        await tqdm_asyncio.gather(
            *(diversify_batch(batch_idx) for batch_idx in range(total_batches)),
            desc="Diversifying questions",
        )
        # Fill any unanswered entries with the original template text to prevent gaps.
        for spec in question_specs:
            spec.setdefault("query", spec["template"])
    else:
        # Use original templates
        for q_spec in question_specs:
            q_spec["query"] = q_spec["template"]
    
    # Generate final question list
    for idx, q_spec in enumerate(question_specs):
        questions.append({
            "qid": f"synth_{task_id}_{idx}",
            "query": q_spec.get("query", q_spec["template"]),
            "answer": q_spec["answer"],
            "position": position,  # All questions are based on complete context
            "category": q_spec["category"]
        })
    
    return questions


async def async_main(
    model: str = DEFAULT_MODEL,
    out: str = "processed_synth.json",
    num_sessions: int = 50,
    num_questions: int = 100,
    no_diversify: bool = False,
    min_turns: int = 20,
    max_turns: int = 50,
    min_scenes: int = 1,
    max_scenes: int = 10,
    fast: bool = False,
    num_records: int = 1,
    llm_concurrency: int = 8,
    client=None,
):
    """Generate one or more synthetic expense-tracking datasets asynchronously."""

    if fast:
        num_sessions = 3
        num_questions = 10
        min_turns = 5
        max_turns = 10
        min_scenes = 1
        max_scenes = 3
        no_diversify = True
        print("🚀 Fast mode enabled: 3 sessions, 10 questions, 5-10 turns, 1-3 scenes, no diversify")

    if out:
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    if num_records < 1:
        raise ValueError("num_records must be >= 1")
    if llm_concurrency < 1:
        raise ValueError("llm_concurrency must be >= 1")
    if num_sessions < 2:
        raise ValueError("num_sessions must be >= 2 (required for question generation)")

    client = client or get_client()

    global _CALL_LLM_SEMAPHORE
    _CALL_LLM_SEMAPHORE = asyncio.Semaphore(llm_concurrency)

    async def build_dataset(dataset_idx: int) -> Dict[str, Any]:
        prefix = ""
        if num_records > 1:
            prefix = f"[Dataset {dataset_idx + 1}/{num_records}] "
            print(f"\n=== Generating dataset {dataset_idx + 1}/{num_records} ===")

        print(f"{prefix}Generating time points...")
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = generate_dates(start_date, end_date, num_sessions)
        print(f"{prefix}Generated {len(dates)} dates: {dates[0]} to {dates[-1]}")

        print(f"{prefix}Generating sessions...")
        all_turns: List[Dict[str, Any]] = []
        all_transactions: List[Transaction] = []
        session_turns: List[List[Dict[str, str]]] = []

        bar_position = dataset_idx

        async def process_session(idx: int, date: str) -> Tuple[int, List[Dict[str, Any]], List[Transaction]]:
            turns, transactions = await generate_session(
                client,
                model,
                date,
                idx,
                is_first_session=(idx == 0),
                min_turns=min_turns,
                max_turns=max_turns,
                min_scenes=min_scenes,
                max_scenes=max_scenes,
            )
            for turn in turns:
                turn["date"] = date
                turn["session_index"] = idx
            return idx, turns, transactions

        session_results = await tqdm_asyncio.gather(
            *(process_session(idx, date) for idx, date in enumerate(dates)),
            desc=f"{prefix}Sessions",
            position=bar_position,
        )

        for _, turns, transactions in sorted(session_results, key=lambda x: x[0]):
            session_turns.append(turns)
            all_turns.extend(turns)
            all_transactions.extend(transactions)

        print(f"{prefix}Total: {len(all_turns)} turns, {len(all_transactions)} transactions")

        print(f"{prefix}Building context...")
        chunks = []
        for turns in tqdm(
            session_turns,
            desc=f"{prefix}Building chunks",
            position=bar_position,
            leave=False,
        ):
            chunk_lines = []
            for turn in turns:
                role_name = "User" if turn["role"] == "user" else "Assistant"
                line = f"[{turn['date']}] {role_name} said, \"{turn['content']}\""
                chunk_lines.append(line)
            chunks.append("\n".join(chunk_lines))

        # 从transactions中提取实际有交易的日期集合
        actual_dates = sorted(list(set(tx.date for tx in all_transactions)))
        print(f"{prefix}Actual dates with transactions: {len(actual_dates)} out of {len(dates)}")

        print(f"{prefix}Generating questions...")
        task_id = "consumption_tracking_" + uuid.uuid4().hex[:8]
        questions = await generate_questions(
            client,
            model,
            all_transactions,
            actual_dates,  # 使用实际有交易的日期而非原始dates
            diversify=not no_diversify,
            position=len(chunks) - 1,
        )
        print(f"{prefix}Generated {len(questions)} questions (target: 36 per sample)")

        transaction_events = []
        for tx in all_transactions:
            transaction_events.append(
                {
                    "date": tx.date,
                    "scene": tx.scene,
                    "subscene": tx.subscene,
                    "amount": tx.amount,
                    "description": tx.description,
                }
            )

        output = {
            "task_id": task_id,
            "questions": questions,
            "chunks": chunks,
            "metadata": {
                "num_sessions": len(dates),
                "num_turns": len(all_turns),
                "num_transactions": len(all_transactions),
                "date_range": {"start": dates[0], "end": dates[-1]},
                "scenes": list(CONSUMPTION_SCENES.keys()),
                "total_amount": sum(tx.amount for tx in all_transactions),
                "transaction_events": transaction_events,
            },
        }

        print(f"{prefix}   - Task ID: {task_id}")
        print(f"{prefix}   - Sessions: {len(dates)}")
        print(f"{prefix}   - Chunks: {len(chunks)}")
        print(f"{prefix}   - Turns: {len(all_turns)}")
        print(f"{prefix}   - Transactions: {len(all_transactions)}")
        print(f"{prefix}   - Questions: {len(questions)}")
        print(f"{prefix}   - Total amount: {sum(tx.amount for tx in all_transactions):.2f}")

        return output

    # Generate samples until we have enough questions
    outputs = []
    total_questions = 0
    sample_idx = 0

    while total_questions < num_questions:
        print(f"\n--- Generating sample {sample_idx + 1} (current total: {total_questions}/{num_questions} questions) ---")
        output = await build_dataset(sample_idx)
        outputs.append(output)

        # Count questions in this sample
        num_q = len(output["questions"])
        total_questions += num_q
        sample_idx += 1

        print(f"Sample {sample_idx} generated {num_q} questions. Total: {total_questions}/{num_questions}")

        # Safety limit to prevent infinite loop
        if sample_idx >= 70:
            print(f"⚠️ Warning: Reached maximum 70 samples. Stopping generation.")
            break

    print(f"\n✅ Generated {sample_idx} samples with {total_questions} total questions (target: {num_questions})")

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
        print(f"✅ Dataset written to: {out}")
    else:
        print("⚠️ Output path not provided. Dataset was generated but not saved.")

    return outputs

def main(
    model: str = DEFAULT_MODEL,
    out: str = "processed_synth.json",
    num_sessions: int = 50,
    num_questions: int = 100,
    no_diversify: bool = False,
    min_turns: int = 20,
    max_turns: int = 50,
    min_scenes: int = 1,
    max_scenes: int = 10,
    fast: bool = False,
    llm_concurrency: int = 8,
    client=None,
):
    """
    Fire entry point that runs the async pipeline.

    num_questions: Target number of questions to generate.
                   Will keep generating samples until reaching this target.
                   Each sample generates up to 36 questions (Type 1-7: 5 each, Type 8: 1).
    """
    print(f"Target: {num_questions} questions")
    print(f"Will generate samples until target is reached (up to 36 questions per sample)")

    return asyncio.run(
        async_main(
            model=model,
            out=out,
            num_sessions=num_sessions,
            num_questions=num_questions,
            no_diversify=no_diversify,
            min_turns=min_turns,
            max_turns=max_turns,
            min_scenes=min_scenes,
            max_scenes=max_scenes,
            fast=fast,
            num_records=1,  # Not used anymore, but keep for compatibility
            llm_concurrency=llm_concurrency,
            client=client,
        )
    )


if __name__ == "__main__":
    if fire is None:
        raise ImportError("Please install the 'fire' package to use the CLI: pip install fire")
    fire.Fire(main)