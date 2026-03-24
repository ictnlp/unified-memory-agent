# locomo prompt experiment

这是一个让 Codex 自动迭代 `locomo` 上 agent prompt 的实验。目标不是改代码逻辑，而是只通过修改 `prompt_template.yaml` 来提升 `llm_score` 对应的准确率。

这是一个长时间运行的自治任务，但当前采用“外层脚本循环重启 Codex，会话内只做一个 step”的执行方式。也就是说，整体任务会持续很久，但单次 Codex 会话不需要无限循环。

`start_codex.sh` 会在外层反复拉起新的 Codex 会话，因此你在单次会话中的职责是：完成一个最小完整工作单元，然后退出，让下一次会话从归档状态继续。

执行要求：

- 单次会话只做一个完整 step。
- 如果这是首次运行，则这个 step 就是基线。
- 如果已经有历史状态，则这个 step 就是一轮新的 prompt 改进实验。
- 完成这个 step 后可以正常退出，由外层脚本再次启动下一次会话。
- 不要在单次会话中连续跑很多轮；否则上下文会越来越大，反而更容易中途退出。

## 设置

开始一轮新实验时，按下面流程初始化：

1. 进入工作目录：
```bash
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified
source ./.venv/bin/activate
```
2. 读取以下文件以获得完整上下文：
   - `run_generation.sh`
   - `run_score.sh`
   - `prompt_template.yaml`
   - `external/verl/memagent/hotpotqa.py`
   - `external/verl/memagent/tool_config.yaml`
3. 先核对 `run_generation.sh` 里的 `PROMPT_TEMPLATE_PATH`。版本控制的目标文件必须是脚本实际读取的 prompt 文件，而不是名字看起来像对的文件。当前脚本如果指向别处，则以脚本里的真实路径为准；如果你本来想优化 `./prompt_template.yaml`，那就先让人工确认路径策略。
4. 只允许修改“脚本实际读取的那个 prompt 文件”。不要改动其它代码、脚本、工具定义和评测逻辑。
5. 检查两个本地记录文件：
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
6. `latest_run` 不单独存文件，而是通过 `prompt_results.tsv` 的最后一条记录推导。

核心要求是：任何一行结果都必须能反查到对应 `run_id`、对应 prompt 快照、对应生成结果和对应评测文件。

**简洁性标准**：在其他条件相同的情况下，越简单越好。一个小幅提升却依赖更长、更绕、更重的提示词结构，通常不值得保留。优先选择那些用更少规则、更短说明就能稳定约束模型行为的改动，例如：

- 直接规定什么时候必须检索
- 直接规定搜索词应该怎样改写
- 直接规定遇到候选条目后必须继续 `memory_key_retrieve`

尽量避免主要引入下面这些复杂度：

- 大段元说明或长篇策略口号
- 层层嵌套的例外分支
- 试图通过很多抽象原则逼出正确行为
- 为了修一个很窄的 bad case，加入很泛化、很冗长的 prompt 结构

判断是否保留某个改动时，要权衡“收益幅度”和“复杂度成本”。一个只带来很小提升、却让 prompt 明显更长、更绕的改动，大概率不值得保留。一个通过更短、更简单的规则拿到持平或更好的结果，这是很好的结果，属于“简化获胜”。

## 输出文件管理

当前工作区已经清理过，可以默认从空状态开始。

后续每一轮运行都允许脚本写入固定文件名：

- `ANALYSIS.md`
- `results/qwen3-4b/locomo/responses_UMA_base_onlymemory.jsonl`
- `results/qwen3-4b/locomo/evaluated_UMA_base_onlymemory.jsonl`

这些文件在工作区中只作为“当前轮的临时产物”存在。真正的长期留档统一依赖 `tmp/prompt_runs/<RUN_ID>/`。

因此每轮实验结束后，都要立即把当前产物移动到对应的归档目录，再开始下一轮修改。

特别注意：

- 旧的 `responses_UMA_base_onlymemory.jsonl` 和 `evaluated_UMA_base_onlymemory.jsonl` 不要留在原位置。
- 如果这些旧文件留在 `results/qwen3-4b/locomo/` 下面，下一轮执行时可能会把它们误当成当前轮结果，导致流程短路或跳过必要步骤。
- 正确做法是：每轮归档时把结果文件从工作区原位置移入 `tmp/prompt_runs/<RUN_ID>/`，确保原位置只保留当前正在运行这一轮生成出来的文件。

## 结果记录格式

`prompt_results.tsv` 使用制表符分隔，表头如下：

```text
run_id	parent_run	accuracy	status	description
```

字段含义：

1. `run_id`：本轮运行唯一标识，格式为 `时间戳_简短说明`
2. `parent_run`：本轮实验实际继承的工作底稿 `run_id`。默认写当前 `best_run`；基线轮填 `baseline`
3. `accuracy`：按 `evaluated_*.jsonl` 中的 `llm_score` 统计出的准确率，保留 4 位小数
4. `status`：`keep`、`discard` 或 `crash`
5. `description`：本轮 prompt 修改意图的简短说明

补充说明：

- `latest_run` 通过读取 `prompt_results.tsv` 的最后一条记录得到，不需要单独增加一列。
- 在后续迭代中，`parent_run` 默认应该等于 `best_run`，因为新实验的工作底稿默认来自当前最优版本，而不是最近一次版本。

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

注意：下面描述的是“单次 Codex 会话应完成的一个 step”，不是会话内无限循环。整体循环由外层 `start_codex.sh` 负责。

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
5. 将当前 prompt、`ANALYSIS.md` 和结果文件移动到 `tmp/prompt_runs/<RUN_ID>/`。
6. 统计 `llm_score` 准确率，写入 `prompt_results.tsv`。
7. 把该 `run_id` 写入 `best_run.txt`。
8. 这次会话的工作到此结束，可以正常退出。外层脚本会继续拉起下一次会话。

### Resume

满足以下任一条件时，优先视为 resume：

- `best_run.txt` 已存在
- `prompt_results.tsv` 已存在
- `tmp/prompt_runs/` 下已经有历史运行目录

resume 时不要重做基线，而是显式区分三个概念：

- `best_run`：当前最高分版本，也是默认工作底稿
- `latest_run`：最近完成的一轮，无论其状态是 `keep`、`discard` 还是 `crash`
- `analysis_target_run`：默认等于 `latest_run`

1. 读取 `best_run.txt`，得到 `<best_run>`。
2. 从 `prompt_results.tsv` 的最后一条记录读取 `<latest_run>`。
3. 默认令 `analysis_target_run=<latest_run>`。
4. 检查 `tmp/prompt_runs/<best_run>/` 是否存在。
5. 如果该目录存在，则将 prompt 文件恢复到该目录中的快照：
```bash
cp "tmp/prompt_runs/<best_run>/prompt_template.yaml" "$PROMPT_FILE"
```
6. 如果 `best_run.txt` 丢失，但 `tmp/prompt_runs/` 下有历史目录，则从 `prompt_results.tsv` 中找最后一个 `status=keep` 的 `run_id`，写回 `best_run.txt`，然后继续。
7. 如果 `best_run.txt` 和 `prompt_results.tsv` 都丢失，但 `tmp/prompt_runs/` 还在，则暂停自动迭代，先人工确认应该从哪个历史目录恢复。

之后执行一轮新的改进实验：

1. 将 prompt 文件恢复到当前 `best_run` 对应的快照，作为本轮唯一工作底稿：
```bash
cp "tmp/prompt_runs/<best_run>/prompt_template.yaml" "$PROMPT_FILE"
```
2. 分析 `analysis_target_run` 的 bad case、评测结果和 `intermediate_paths` 对应的中间文件。
3. 形成或更新 `ANALYSIS.md`。文档中必须同时写清楚：
   - `Current best run`
   - `Latest analyzed run`
   - `Observed issues in latest run`
   - `What to keep from best`
   - `What to borrow or reject from latest`
   - `Planned delta for next prompt`
4. 如果 `latest_run == best_run`，则分析对象和工作底稿天然一致，直接从当前 best 继续改。
5. 如果 `latest_run != best_run`，则执行固定策略：`analyze latest, edit from best`。
   - 分析对象仍然是 `latest_run`
   - 工作底稿仍然是 `best_run`
   - 实际改动方式是：只把 `latest_run` 中看起来有价值的局部想法吸收到 `best_run` prompt 中
6. 明确禁止两种模糊行为：
   - 不要“分析 latest，但直接在 latest prompt 上继续改”
   - 不要“恢复 best 后，完全忽略 latest 的失败信号”
7. 只修改 prompt 文件，不要改其它文件。
8. 生成新的 `run_id`，运行生成和评测。
9. 将本轮 `ANALYSIS.md`、prompt 快照和结果文件移动到 `tmp/prompt_runs/<RUN_ID>/`。
10. 统计准确率，记录到 `prompt_results.tsv`。
11. 如果分数优于当前最优版本，则：
   - 将该轮记为 `keep`
   - 用当前 `run_id` 更新 `best_run.txt`
12. 如果分数持平或更差，则：
   - 将该轮记为 `discard`
   - 不需要在本次会话中继续下一轮；下次会话开始时会再次从 `best_run.txt` 恢复
13. 如果运行失败、输出缺失或评测崩溃，则记为 `crash`，准确率填 `0.0000`。失败分析仍然可以参考 `latest_run`，但下次会话开始时依旧从 `best_run` 恢复。
14. 记录完成后，本次会话结束。外层脚本会启动下一次会话。

### 何时退出本次会话

以下情况都可以结束当前这次 Codex 会话：

- 已完成一个完整 step，并且已经归档和记录结果
- 缺少必要输入文件，且无法从现有归档恢复
- 关键依赖服务无法启动，例如 vLLM 启动失败
- 历史状态损坏，无法判断应该从哪里 resume，需要人工介入

如果是异常退出，必须在日志或输出中明确写出阻塞原因，便于下一次会话或人工处理。

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
