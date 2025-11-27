import json
import os
import random
os.makedirs("small_valsets", exist_ok=True)

# for bench in ["synth-s1", "synth-s3", "synth-s50", "banking77", "booksum", "clinic", "hotpotqa_200", "locomo", "longmemeval", "msc", "nlu", "perltqa", "pubmed_rct", "trec_coarse", "trec_fine", "squad", "infbench"]:
for bench in ["synth-s10"]:
    file = f"processed_{bench}.json"
    data = json.load(open(file, "r"))
    small_valset = []

    num_q_sofar = 0
    for sample in data:
        if num_q_sofar + len(sample["questions"]) >= 10:
            # sample["questions"] = sample["questions"][: 10 - num_q_sofar]
            sample["questions"] = random.sample(sample["questions"], 10 - num_q_sofar)
            small_valset.append(sample)
            break
        else:
            small_valset.append(sample)
            num_q_sofar += len(sample["questions"])
    with open(f"small_valsets/processed_{bench}.json", "w") as f:
        json.dump(small_valset, f, indent=4, ensure_ascii=False)