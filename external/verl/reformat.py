# reformat
import json
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--task", type=str, required=True, choices=["longmemeval", "hotpotqa", "locomo", "msc"])
parser.add_argument("--length", type=int, default=200, choices=[50, 100, 200, 400, 800, 1600, 3200, 6400])
args = parser.parse_args()

input_file = args.input_file
output_file = input_file.replace(".jsonl", "_reformat.jsonl")
data = [json.loads(line) for line in open(input_file)]

if args.task == "longmemeval":
    original_file = "/mnt/pfs-guan-ssai/nlu/zhangkehao/longmemeval_s.json"
    original_data = json.load(open(original_file))
elif args.task == "hotpotqa":
    DATAROOT = "/mnt/pfs-guan-ssai/nlu/zhangkehao/MemAgent_minimal/data/taskutils/memory_data"
    original_data = load_dataset("json", data_files=f"{DATAROOT}/eval_{args.length}.json", split="train")
elif args.task == "locomo":
    original_file = "/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/processed_locomo.json"
    original_data = json.load(open(original_file))
elif args.task == "msc":
    original_file = "/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/processed_msc_batch5.json"
    original_data = json.load(open(original_file))

with open(output_file, "w") as f:
    for d in data:
        if args.task == "locomo" or args.task == "msc":
            sample_id, question_id = d['idx'].split('_')
            sample_id = int(sample_id)
            question_id = int(question_id)
            original_sample = original_data[sample_id]
            original_d = original_sample['questions'][question_id]
        else:
            original_d = original_data[d['idx']]
        
        response = d["response"]
        if "\\boxed{" in response and "}" in response.split("\\boxed{")[-1]:
            response = response.split("\\boxed{")[-1]
            response = response[:response.rindex("}")]
        else:
            response = response[-200:]
        
        if args.task == "longmemeval":
            new_d = {
                "qid": "longmemeval_" + original_d["question_id"] + "_0",
                "query": original_d["question"],
                "expected_answer": original_d["answer"],
                "response": response
            }
        elif args.task == "hotpotqa":
            assert len(original_d["answers"]) == 1
            new_d = {
                "qid": f"hotpotqa_{args.length}_{original_d['index']}_0",
                "query": original_d["input"],
                "expected_answer": original_d["answers"][0],
                "response": response
            }
        else:  # locomo
            new_d = {
                "qid": original_d["qid"],
                "query": original_d["query"],
                "expected_answer": original_d["answer"],
                "response": response
            }
        f.write(json.dumps(new_d, ensure_ascii=False) + '\n')