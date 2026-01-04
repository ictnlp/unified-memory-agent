import json
import os
import random
os.makedirs("small_valsets", exist_ok=True)

# for bench in ["synth-s1", "synth-s3", "synth-s50", "banking77", "booksum", "clinic", "hotpotqa_200", "locomo", "longmemeval", "msc_batch5", "nlu", "perltqa", "pubmed_rct", "trec_coarse", "trec_fine", "squad", "infbench", "convomem"]:
for bench in ["synth-ss2", "synth-ss3", "synth-ss4", "synth-ss5", "synth-ss10", "synth-ss20", "synth-ss30", "synth-ss40", "synth-ss50"]:
    file = f"processed_{bench}.json"
    data = json.load(open(file, "r"))
    random.shuffle(data)  # 随机打乱样本顺序
    small_valset = []

    num_q_sofar = 0
    expected_num_q = 10
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