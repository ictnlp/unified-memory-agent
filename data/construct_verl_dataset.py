import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def load_samples(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("Expected input JSON to be a list of samples")
    return samples


def normalize_chunks(chunks: Sequence[str]) -> str:
    normalized_parts: List[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            normalized_parts.append(chunk)
            continue
        if chunk.startswith("Below is a conversation between"):
            newline_pos = chunk.find("\n")
            if newline_pos != -1:
                normalized_parts.append(chunk[newline_pos + 1 :])
                continue
        normalized_parts.append(chunk)
    return "\n\n".join(normalized_parts)


def build_tools_kwargs(chunks: Sequence[str], memory_store: Path) -> Dict[str, Dict]:
    chunks_array = np.array(chunks, dtype=object)
    filename = str(memory_store)
    return {
        "memory_add": {"create_kwargs": {"filename": filename}},
        "memory_bm25_retrieve": {"create_kwargs": {"chunks": chunks_array}},
        "memory_delete": {"create_kwargs": {"filename": filename}},
        "memory_embedding_retrieve": {"create_kwargs": {"chunks": chunks_array}},
        "memory_key_retrieve": {"create_kwargs": {"filename": filename}},
        "memory_list": {"create_kwargs": {"filename": filename}},
        "memory_update": {"create_kwargs": {"filename": filename}},
    }


def construct_dataset(
    samples: Sequence[Dict],
    dataset_name: str,
    memory_store: Path,
    batch_size: int,
    agent_name: str,
) -> pd.DataFrame:
    data = {
        "data_source": [],
        "prompt": [],
        "context": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }

    for sample_index, sample in enumerate(samples):
        questions = sample.get("questions", [])
        chunks = sample.get("chunks", [])
        if not chunks:
            continue

        context_text = normalize_chunks(chunks)
        tools_kwargs = build_tools_kwargs(chunks, memory_store)
        chunk_array = np.array(chunks, dtype=object)

        for batch_start in range(0, len(questions), batch_size):
            batch_questions = questions[batch_start : batch_start + batch_size]
            if not batch_questions:
                continue

            question_texts = [q.get("query", "") for q in batch_questions]
            answers = [q.get("answer", "") for q in batch_questions]

            prompt_item = {"content": question_texts, "role": "user"}
            reward_model = {"ground_truth": answers}

            extra_info = {
                "task_id": sample.get("task_id"),
                "sample_index": sample_index,
                "batch_index": batch_start // batch_size,
                "num_chunks": len(chunks),
                "question_ids": [q.get("qid") for q in batch_questions],
                "question": question_texts,
                "tools_kwargs": tools_kwargs,
                "chunks": chunk_array,
            }

            data["data_source"].append(dataset_name)
            data["prompt"].append(np.array([prompt_item], dtype=object))
            data["context"].append(context_text)
            data["reward_model"].append(reward_model)
            data["extra_info"].append(extra_info)
            data["agent_name"].append(agent_name)

    if not data["data_source"]:
        raise ValueError("No samples produced; check input data")

    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct VERL training dataset from annotated samples")
    parser.add_argument("input", type=Path, help="Path to JSON dataset (list of samples)")
    parser.add_argument("output", type=Path, help="Path to write parquet dataset")
    parser.add_argument(
        "--memory-store",
        type=Path,
        default=Path("/mnt/pfs-guan-ssai/nlu/zhangkehao/verl/memagent/store/memory_store.jsonl"),
        help="Memory store file used by memory tools",
    )
    parser.add_argument("--dataset-name", default="generic", help="Dataset identifier written to data_source column")
    parser.add_argument("--agent-name", default="tool_mem_agent", help="Agent name stored in dataset")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of questions per training example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input)
    df = construct_dataset(
        samples=samples,
        dataset_name=args.dataset_name,
        memory_store=args.memory_store,
        batch_size=args.batch_size,
        agent_name=args.agent_name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"Processed {len(samples)} input samples -> {len(df)} training examples")
    print(f"Saved dataset to: {args.output}")


if __name__ == "__main__":
    main()
