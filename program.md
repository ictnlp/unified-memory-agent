# locomo prompt experiment

这是一个让 Codex 自动迭代 `locomo` 上 agent prompt 的实验。目标不是改代码逻辑，而是只通过修改 `prompt_template.yaml` 来提升 `llm_score` 对应的准确率。

## 设置

开始一轮新实验时，按下面流程初始化：

1. 进入工作目录：
```bash
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified
source ./.venv/bin/activate
```
2. 检查当前仓库状态：
```bash
git status --short --branch
```
3. 创建一个新的实验分支，命名建议为 `locomo-prompt/<tag>`，其中 `<tag>` 用当天日期或自定义运行标签，例如 `locomo-prompt/20260323a`。
4. 读取以下文件以获得完整上下文：
   - `run_generation.sh`
   - `run_score.sh`
   - `prompt_template.yaml`
   - `external/verl/memagent/hotpotqa.py`
   - `external/verl/memagent/tool_config.yaml`
5. 先核对 `run_generation.sh` 里的 `PROMPT_TEMPLATE_PATH`。版本控制的目标文件必须是脚本实际读取的 prompt 文件，而不是名字看起来像对的文件。当前脚本如果指向别处，则以脚本里的真实路径为准；如果你本来想优化 `./prompt_template.yaml`，那就先让人工确认路径策略。
6. 只允许修改“脚本实际读取的那个 prompt 文件”。不要改动其它代码、脚本、工具定义和评测逻辑。
7. 初始化两个未跟踪文件：
   - `prompt_results.tsv`
   - `best_prompt_commit.txt`

## 版本控制原则

这类实验的关键不是“有没有 commit”，而是“每个分数能不能追溯到唯一 prompt 版本和唯一输出文件”。

采用以下规则：

1. `git commit` 是 prompt 版本的唯一正式编号。
2. 每次实验都必须先提交，再运行。不要在未提交状态下直接跑实验。
3. 记住一个变量：`PROMPT_FILE=<脚本实际读取的 prompt 文件路径>`。
4. 每次运行都生成一个 `run_id`，格式建议为：
```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"
```
5. 每次运行结束后，把本轮使用的 prompt 和产物一起归档到：
```text
./tmp/prompt_runs/<RUN_ID>/
```
目录下至少保留：
   - `prompt_template.yaml`
   - `ANALYSIS.md`
   - `responses_UMA_base_onlymemory.jsonl`
   - `evaluated_UMA_base_onlymemory.jsonl`
6. `prompt_results.tsv` 负责记录实验账本，归档目录负责保存证据，git commit 负责保存版本。
7. `best_prompt_commit.txt` 里只保存当前最优版本的短 commit hash，作为后续迭代的基线。

核心要求是：任何一行结果都必须能反查到对应 commit、对应 prompt 快照、对应生成结果和对应评测文件。

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
run_id	commit	parent_commit	accuracy	status	description
```

字段含义：

1. `run_id`：本轮运行唯一标识，格式为 `时间戳_短commit`
2. `commit`：当前实验版本的短 commit hash
3. `parent_commit`：本轮实验实际继承的最优版本 short hash；基线轮填 `baseline`
4. `accuracy`：按 `evaluated_*.jsonl` 中的 `llm_score` 统计出的准确率，保留 4 位小数
5. `status`：`keep`、`discard` 或 `crash`
6. `description`：本轮 prompt 修改意图的简短说明

示例：

```text
run_id	commit	parent_commit	accuracy	status	description
20260323T060000Z_a1b2c3d	a1b2c3d	baseline	0.5420	keep	initial prompt baseline
20260323T071500Z_b2c3d4e	b2c3d4e	a1b2c3d	0.5710	keep	clarify memory retrieval priority
20260323T083000Z_c3d4e5f	c3d4e5f	b2c3d4e	0.5630	discard	add verbose tool-use instructions
20260323T094500Z_d4e5f6g	d4e5f6g	b2c3d4e	0.0000	crash	force unsupported output format
```

## 实验循环

第一次运行必须始终是基线：

1. 保持当前“脚本实际读取的 prompt 文件”不变，先提交一次基线 commit。
2. 运行生成：
```bash
bash run_generation.sh
```
3. 运行评测：
```bash
bash run_score.sh
```
4. 统计 `llm_score` 准确率，写入 `prompt_results.tsv`。
5. 把该 commit 写入 `best_prompt_commit.txt`。

之后进入无限循环：

1. 读取 `best_prompt_commit.txt`，将 prompt 文件恢复到当前最优版本：
```bash
git restore --source <best_commit> -- "$PROMPT_FILE"
```
2. 分析最近一次评测结果和 bad case，重点阅读 `intermediate_paths` 对应的中间文件。
3. 形成或更新 `ANALYSIS.md`，总结当前问题、失败模式和下一轮 prompt 修改假设。
4. 只修改 prompt 文件，不要改其它文件。
5. 提交本轮 prompt 版本：
```bash
git add "$PROMPT_FILE"
git commit -m "prompt: <short description>"
```
6. 生成 `run_id`，运行生成和评测。
7. 将本轮 `ANALYSIS.md`、prompt 快照和结果文件拷贝到 `tmp/prompt_runs/<RUN_ID>/`。
8. 统计准确率，记录到 `prompt_results.tsv`。
9. 如果分数优于当前最优版本，则：
   - 将该轮记为 `keep`
   - 用当前 commit 更新 `best_prompt_commit.txt`
10. 如果分数持平或更差，则：
   - 将该轮记为 `discard`
   - 不要 reset 整个仓库
   - 下一轮开始前仅恢复 prompt 文件回到最优版本
11. 如果运行失败、输出缺失或评测崩溃，则记为 `crash`，准确率填 `0.0000`，然后回到最优 prompt 继续。

## 为什么不用 reset 整个分支

这里不建议频繁 `git reset --hard`，原因有两个：

1. 这个仓库可能同时存在其它未提交改动，硬回退容易误伤用户文件。
2. 对 prompt 搜索来说，保留全部失败尝试的 commit 历史是有价值的，后面可以复盘哪些方向无效。

因此这里采用：

- `best_prompt_commit.txt` 记录当前最优版本
- 所有试验 commit 都保留
- 工作区只恢复 prompt 文件，不回滚整个仓库

这是一种更稳妥的“最佳版本推进”机制。

## 约束

- 不要修改 `run_generation.sh`、`run_score.sh`、`evaluate_async.py`、工具定义文件或任何 Python 代码。
- 不要假设 `./prompt_template.yaml` 一定是生效文件，先以 `PROMPT_TEMPLATE_PATH` 为准。
- 不要新增依赖，不要改模型，不要改 judge。
- 不要跳过基线。
- 不要只看总体分数，要结合 bad case 和 `intermediate_paths` 做分析。
- 不要在未 commit 的状态下运行实验。
- 不要把 `prompt_results.tsv`、`tmp/prompt_runs/` 下的产物提交到 git，保持它们为本地实验记录。

## 成功标准

唯一目标是把 `locomo` 上基于 `llm_score` 统计的准确率做高。只要一个 prompt 修改被证明更好，就把它升级为新的最优版本，并从它继续向前搜索。
