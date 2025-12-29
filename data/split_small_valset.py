import json
import os
import random
os.makedirs("small_valsets", exist_ok=True)

# for bench in ["synth-s1", "synth-s3", "synth-s50", "banking77", "booksum", "clinic", "hotpotqa_200", "locomo", "longmemeval", "msc_batch5", "nlu", "perltqa", "pubmed_rct", "trec_coarse", "trec_fine", "squad", "infbench", "convomem"]:
for bench in ["synth-ss10_train"]:
    file = f"processed_{bench}.json"
    data = json.load(open(file, "r"))
    random.shuffle(data)  # 随机打乱样本顺序
    small_valset = []

    num_q_sofar = 0
    expected_num_q = 300
    for sample in data:
        if num_q_sofar + len(sample["questions"]) >= expected_num_q:
            sample["questions"] = random.sample(sample["questions"], expected_num_q - num_q_sofar)
            small_valset.append(sample)
            break
        else:
            small_valset.append(sample)
            num_q_sofar += len(sample["questions"])
    with open(f"small_valsets/processed_{bench}.json", "w") as f:
        json.dump(small_valset, f, indent=4, ensure_ascii=False)