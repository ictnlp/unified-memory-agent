# AmemAgent 使用文档

## 简介

AmemAgent 是对 A-mem (Agentic Memory) 系统的完整封装，支持动态记忆演化和智能检索。

论文：[A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

## 核心特性

- ✅ **动态记忆演化**：基于 Zettelkasten 原理，自动建立记忆之间的链接
- ✅ **智能元数据提取**：LLM 自动提取 keywords, context, tags
- ✅ **向量检索**：ChromaDB 进行语义相似度搜索
- ✅ **原始实现**：使用官方 A-mem 代码，确保科研严谨性

## 使用方法

### 1. 基本评估

```bash
# 在 Locomo 数据集上评估 AmemAgent
python evaluate_async.py --task locomo --agent amem --concurrency 10

# 在 LongmemEval 数据集上评估
python evaluate_async.py --task longmemeval --agent amem --concurrency 10

# 使用不同模型
python evaluate_async.py --task locomo --agent amem --model gpt4.1
```

### 2. 配置选项

#### 并发控制
```bash
# 低并发（更安全，避免 API rate limit）
python evaluate_async.py --task locomo --agent amem --concurrency 5

# 高并发（更快，但需要确保 API 限额）
python evaluate_async.py --task locomo --agent amem --concurrency 50
```

#### 输出目录
```bash
# 自定义输出目录
python evaluate_async.py --task locomo --agent amem --output_dir my_results
```

### 3. 代码中使用

```python
from openai import AsyncOpenAI
from agents.amem_agent import AmemAgent

# 创建客户端
client = AsyncOpenAI(
    api_key="your-key",
    base_url="your-endpoint"
)

# 创建 AmemAgent（默认开启演化）
agent = AmemAgent(
    client=client,
    model_name="gpt4.1",
    enable_evolution=True,  # 默认开启
    evo_threshold=100,       # 每 100 个记忆触发一次演化
)

# 添加记忆
await agent.add_memory_async("Alice lives in New York.")
await agent.add_memory_async("Bob works at Google.")

# 批量问答
questions = ["Where does Alice live?", "What does Bob do?"]
answers = await agent.QA_batch_async(questions)

# 重置（新 sample）
agent.reset()
```

## 配置参数

### AmemAgent 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `client` | 必填 | OpenAI/AsyncOpenAI 客户端 |
| `model_name` | `"gpt4.1"` | LLM 模型名称 |
| `embedding_model` | `"all-MiniLM-L6-v2"` | SentenceTransformer 模型 |
| `evo_threshold` | `100` | 触发演化的记忆数量阈值 |
| `enable_evolution` | `True` | 是否启用记忆演化 |

### 性能调优

#### 禁用演化（不推荐）
如果只想测试向量检索效果，可以禁用演化：

```python
agent = AmemAgent(
    client=client,
    model_name="gpt4.1",
    enable_evolution=False,  # 禁用演化
)
```

**注意**：禁用演化会失去 A-mem 的核心优势，不建议在正式评估中使用。

#### 调整演化阈值
```python
agent = AmemAgent(
    client=client,
    enable_evolution=True,
    evo_threshold=50,  # 更频繁的演化
)
```

#### 自定义检索数量
```python
agent = AmemAgent(client=client)
agent.top_k_retrieval = 10  # 检索 top-10 记忆（默认 5）
```

## 依赖项

AmemAgent 需要以下额外依赖（已自动安装）：

```
chromadb>=0.4.22
sentence-transformers>=2.2.2
rank_bm25>=0.2.2
nltk>=3.8.1
litellm>=1.16.11
```

## 性能考虑

### 演化开销

- 每次 `add_memory` 可能触发演化（取决于 `evo_threshold`）
- 演化涉及：
  1. 检索 top-5 相关记忆
  2. LLM 分析是否需要演化
  3. 更新记忆链接和标签
- **预估开销**：每次演化约 2-5 秒（1-2 次 LLM 调用）

### 检索开销

- ChromaDB 向量检索：0.01-0.1 秒
- SentenceTransformer embedding：0.05 秒

### 总体性能

对于典型的评估场景（50 chunks, 10 questions）：

- **内存添加**：50 × 0.1秒 = 5秒（无演化）或 50 × 2秒 = 100秒（有演化）
- **问答**：10 × (0.1秒检索 + 10秒LLM) = 101秒
- **总计**：约 106秒（无演化）或 201秒（有演化）

## 输出文件

```
results/locomo/
├── responses_amem_20260127_120000.jsonl     # 生成的回答
├── evaluated_amem_20260127_120000.jsonl     # 评估结果
└── intermediate_outputs/                    # （如果开启）
```

## 对比其他 Baseline

```bash
# Concat Agent（简单拼接）
python evaluate_async.py --task locomo --agent concat

# RAG Agent（静态检索）
python evaluate_async.py --task locomo --agent rag

# A-mem Agent（动态演化）
python evaluate_async.py --task locomo --agent amem
```

## 常见问题

### Q: 演化是否影响评估结果？

A: 是的，演化是 A-mem 的核心特性。论文表明演化能提升长期记忆的组织质量，从而改善检索效果。禁用演化就不是真正的 A-mem 了。

### Q: 如何确认演化是否生效？

A: 检查日志中是否有演化相关的输出，或者查看 ChromaDB 中记忆的 links 字段。

### Q: 为什么比 Concat Agent 慢？

A: AmemAgent 包含：
1. embedding 生成
2. ChromaDB 索引
3. 记忆演化（LLM 调用）
4. 向量检索

这些额外操作换来了更好的长期记忆管理能力。

### Q: ChromaDB 数据存储在哪里？

A: 默认存储在内存中，每次 `reset()` 会清空。如需持久化，需修改 A-mem 配置。

## 调试

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = AmemAgent(...)
```

### 检查记忆状态

```python
# 查看已存储的记忆数量
print(f"Total memories: {len(agent.memory_system.memories)}")

# 查看特定记忆
memory = agent.memory_system.read(memory_id)
print(f"Content: {memory.content}")
print(f"Keywords: {memory.keywords}")
print(f"Links: {memory.links}")
```

## 参考文献

- A-mem 论文：https://arxiv.org/pdf/2502.12110
- A-mem GitHub：https://github.com/agiresearch/A-mem
- ChromaDB：https://docs.trychroma.com/
