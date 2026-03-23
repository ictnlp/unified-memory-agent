# locomo prompt experiment

这是一个让 Codex 自动迭代 `locomo` 上 agent prompt 的实验。目标不是改代码逻辑，而是只通过修改 `prompt_template.yaml` 来提升 `llm_score` 对应的准确率。

## 设置

开始一轮新实验时，按下面流程初始化：

1. 进入工作目录：
```bash
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified
source ./.venv/bin/activate
```
2. 先检查 embedding 服务。
   - 优先检查 `http://localhost:8080/health` 是否可用。
   - 如果服务已经存在，则直接跳过，不要重复启动。
   - 如果服务不存在，则按照 `run_generation.sh` 里注释掉的那条命令启动，并先切换到对应虚拟环境：
```bash
source external/infinity/libs/infinity_emb/.venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
```
   - 启动完 embedding 服务后，再切回本项目虚拟环境：
```bash
source ./.venv/bin/activate
```
3. 读取以下文件以获得完整上下文：
   - `run_generation.sh`
   - `run_score.sh`
   - `prompt_template.yaml`
   - `external/verl/memagent/hotpotqa.py`
   - `external/verl/memagent/tool_config.yaml`
4. 先核对 `run_generation.sh` 里的 `PROMPT_TEMPLATE_PATH`。版本控制的目标文件必须是脚本实际读取的 prompt 文件，而不是名字看起来像对的文件。当前脚本如果指向别处，则以脚本里的真实路径为准；如果你本来想优化 `./prompt_template.yaml`，那就先让人工确认路径策略。
5. 只允许修改“脚本实际读取的那个 prompt 文件”。不要改动其它代码、脚本、工具定义和评测逻辑。
6. 检查两个本地记录文件：
   - `prompt_results.tsv`
   - `best_run.txt`
   如果不存在，则在第一次运行时创建；如果已经存在，则说明这不是第一次运行，应当进入 resume 流程，而不是重做基线。

## 版本控制原则

这类实验不使用 git 维护版本。原因是当前磁盘是网络挂载，`.git` 容易损坏，因此版本管理完全依赖本地归档目录和结果台账。

关键要求不是“有没有版本控制工具”，而是“每个分数能不能追溯到唯一 prompt 版本和唯一输出文件”。

采用以下规则：

1. 记住一个变量：`PROMPT_FILE=<脚本实际读取的 prompt 文件路径>`。
2. 每次运行都生成一个带时间前缀的 `run_id`，格式建议为：
```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_<short_label>"
```
例如：
```text
20260323T060000Z_baseline
20260323T071500Z_memory_priority
20260323T083000Z_shorter_tool_rules
```
这样的命名天然按时间排序，直接 `ls` 就能看出先后顺序。
3. 每次运行结束后，把本轮使用的 prompt 和产物一起归档到：
```text
./tmp/prompt_runs/<RUN_ID>/
```
目录下至少保留：
   - `prompt_template.yaml`
   - `ANALYSIS.md`
   - `responses_UMA_base_onlymemory.jsonl`
   - `evaluated_UMA_base_onlymemory.jsonl`
4. `prompt_results.tsv` 负责记录实验账本，归档目录负责保存证据。
5. `best_run.txt` 里只保存当前最优实验的 `run_id`，作为后续迭代的基线。

核心要求是：任何一行结果都必须能反查到对应 `run_id`、对应 prompt 快照、对应生成结果和对应评测文件。

## 输出文件管理

当前工作区已经清理过，可以默认从空状态开始。

后续每一轮运行都允许脚本写入固定文件名：

- `ANALYSIS.md`
- `results/qwen3-4b/locomo/responses_UMA_base_onlymemory.jsonl`
- `results/qwen3-4b/locomo/evaluated_UMA_base_onlymemory.jsonl`

这些文件在工作区中只保留“最近一轮”的版本即可，不要求每次先重命名旧文件。真正的长期留档统一依赖 `tmp/prompt_runs/<RUN_ID>/`。

因此每轮实验结束后，都要立即把当前产物拷贝到对应的归档目录，再开始下一轮修改。

## 结果记录格式

`prompt_results.tsv` 使用制表符分隔，表头如下：

```text
run_id	parent_run	accuracy	status	description
```

字段含义：

1. `run_id`：本轮运行唯一标识，格式为 `时间戳_简短说明`
2. `parent_run`：本轮实验实际继承的最优实验 `run_id`；基线轮填 `baseline`
3. `accuracy`：按 `evaluated_*.jsonl` 中的 `llm_score` 统计出的准确率，保留 4 位小数
4. `status`：`keep`、`discard` 或 `crash`
5. `description`：本轮 prompt 修改意图的简短说明

示例：

```text
run_id	parent_run	accuracy	status	description
20260323T060000Z_baseline	baseline	0.5420	keep	initial prompt baseline
20260323T071500Z_memory_priority	20260323T060000Z_baseline	0.5710	keep	clarify memory retrieval priority
20260323T083000Z_verbose_tool_rules	20260323T071500Z_memory_priority	0.5630	discard	add verbose tool-use instructions
20260323T094500Z_bad_output_format	20260323T071500Z_memory_priority	0.0000	crash	force unsupported output format
```

## 实验循环

先判断这是首次运行还是继续运行。

一旦进入实验循环，不要在完成一次基线或一次改进后退出。除非遇到明确阻塞条件，否则必须持续进行“分析 -> 修改 prompt -> 生成 -> 评测 -> 归档 -> 记录 -> 进入下一轮”的循环，直到被人工手动停止。

### 首次运行

满足以下条件时，视为首次运行：

- `best_run.txt` 不存在
- `prompt_results.tsv` 不存在
- `tmp/prompt_runs/` 不存在，或者目录为空

首次运行必须始终是基线：

1. 保持当前“脚本实际读取的 prompt 文件”不变，直接运行一次基线。
2. 运行生成：
```bash
bash run_generation.sh
```
3. 运行评测：
```bash
bash run_score.sh
```
4. 生成 `run_id`，建议使用 `20260323T060000Z_baseline` 这种格式。
5. 将当前 prompt、`ANALYSIS.md` 和结果文件归档到 `tmp/prompt_runs/<RUN_ID>/`。
6. 统计 `llm_score` 准确率，写入 `prompt_results.tsv`。
7. 把该 `run_id` 写入 `best_run.txt`。
8. 基线完成后不要退出，立即进入下面的“持续迭代”流程。

### Resume

满足以下任一条件时，优先视为 resume：

- `best_run.txt` 已存在
- `prompt_results.tsv` 已存在
- `tmp/prompt_runs/` 下已经有历史运行目录

resume 时不要重做基线，而是从当前最优实验继续：

1. 读取 `best_run.txt`，得到 `<best_run>`。
2. 检查 `tmp/prompt_runs/<best_run>/` 是否存在。
3. 如果该目录存在，则将 prompt 文件恢复到该目录中的快照：
```bash
cp "tmp/prompt_runs/<best_run>/prompt_template.yaml" "$PROMPT_FILE"
```
4. 如果 `best_run.txt` 丢失，但 `tmp/prompt_runs/` 下有历史目录，则从 `prompt_results.tsv` 中找最后一个 `status=keep` 的 `run_id`，写回 `best_run.txt`，然后继续。
5. 如果 `best_run.txt` 和 `prompt_results.tsv` 都丢失，但 `tmp/prompt_runs/` 还在，则暂停自动迭代，先人工确认应该从哪个历史目录恢复。

之后进入持续迭代：

1. 读取 `best_run.txt`，将 prompt 文件恢复到当前最优实验对应的快照：
```bash
cp "tmp/prompt_runs/<best_run>/prompt_template.yaml" "$PROMPT_FILE"
```
2. 分析最近一次评测结果和 bad case，重点阅读 `intermediate_paths` 对应的中间文件。
3. 形成或更新 `ANALYSIS.md`，总结当前问题、失败模式和下一轮 prompt 修改假设。
4. 只修改 prompt 文件，不要改其它文件。
5. 生成新的 `run_id`，运行生成和评测。
6. 将本轮 `ANALYSIS.md`、prompt 快照和结果文件拷贝到 `tmp/prompt_runs/<RUN_ID>/`。
7. 统计准确率，记录到 `prompt_results.tsv`。
8. 如果分数优于当前最优版本，则：
   - 将该轮记为 `keep`
   - 用当前 `run_id` 更新 `best_run.txt`
9. 如果分数持平或更差，则：
   - 将该轮记为 `discard`
   - 下一轮开始前仅恢复 prompt 文件回到最优版本
10. 如果运行失败、输出缺失或评测崩溃，则记为 `crash`，准确率填 `0.0000`，然后回到最优 prompt 继续。
11. 无论本轮结果是 `keep`、`discard` 还是 `crash`，都不要退出；应当继续开始下一轮。

### 停止条件

只有以下情况允许停止：

- 人工明确要求停止
- 缺少必要输入文件，且无法从现有归档恢复
- 关键依赖服务无法启动，例如 embedding 服务或 vLLM 连续多次启动失败
- 连续多轮都因同一个基础设施问题崩溃，继续重试没有意义

除上述情况外，不要输出总结后退出，不要在完成基线后退出，也不要在完成某一轮改进后等待人工确认。

## 为什么使用时间前缀归档

这里采用时间前缀目录，而不是无语义编号，原因是：

1. 目录名本身就能表达先后顺序，天然可排序。
2. 每轮实验的 prompt、分析和结果都在一个目录里，更容易复盘。
3. 不依赖 `.git`，更适合网络挂载磁盘。

因此这里采用：

- `best_run.txt` 记录当前最优版本
- `tmp/prompt_runs/<RUN_ID>/` 保存每轮完整快照
- 工作区只恢复 prompt 文件，不回滚整个仓库

这是一种更稳妥的“最佳版本推进”机制。

## 约束

- 不要修改 `run_generation.sh`、`run_score.sh`、`evaluate_async.py`、工具定义文件或任何 Python 代码。
- 不要假设 `./prompt_template.yaml` 一定是生效文件，先以 `PROMPT_TEMPLATE_PATH` 为准。
- 不要新增依赖，不要改模型，不要改 judge。
- 不要跳过基线。
- 不要只看总体分数，要结合 bad case 和 `intermediate_paths` 做分析。
- 不要依赖 git 管理实验版本。
- 不要覆盖历史归档目录。

## 成功标准

唯一目标是把 `locomo` 上基于 `llm_score` 统计的准确率做高。只要一个 prompt 修改被证明更好，就把它升级为新的最优版本，并从它继续向前搜索。
