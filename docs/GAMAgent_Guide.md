# GAMAgent 使用指南

## 概述

GAMAgent 是一个基于 [GAM (General Agentic Memory)](https://github.com/gam-framework/gam) 框架实现的高级记忆代理。它结合了 MemoryAgent 进行记忆构建和 ResearchAgent 进行智能问答。

## 特性

- **高级记忆构建**: 使用 GAM 的 MemoryAgent 自动提取和组织记忆摘要
- **智能检索**: 支持多种检索器（Index, BM25, Dense）
- **研究式问答**: 使用 ResearchAgent 进行多轮迭代推理
- **灵活配置**: 支持本地和远程 API、多种检索器组合

## 安装依赖

首先需要安装 GAM 框架：

```bash
# 克隆 GAM 仓库（假设在相邻目录）
git clone https://github.com/your-org/general-agentic-memory.git ../general-agentic-memory

# 安装 GAM 依赖
cd ../general-agentic-memory
pip install -r requirements.txt
cd ../unified-memory-agent
```

## 基本使用

### 1. 导入和初始化

```python
from openai import OpenAI
from config import AGENT_CLASS, API_CONFIG

# 创建 OpenAI client
client = OpenAI(
    base_url=API_CONFIG["base_url"],
    api_key=API_CONFIG["api_key"],
    default_headers=API_CONFIG["default_headers"]
)

# 获取 GAMAgent 类并初始化
GAMAgent = AGENT_CLASS['gam']
agent = GAMAgent(
    client=client,
    model_name="gpt4.1",
    use_local_api=True,  # 使用本地 API
    max_research_iters=3,  # 研究最大迭代次数
    retriever_types=["bm25"],  # 使用的检索器类型
    index_dir="./tmp/gam_indices"  # 索引存储目录
)
```

### 2. 添加记忆

```python
# 添加单个记忆块
agent.add_memory("人工智能是计算机科学的一个分支。")

# 批量添加记忆
documents = [
    "机器学习是 AI 的一个子集。",
    "深度学习使用多层神经网络。",
    "NLP 专注于理解人类语言。"
]

for doc in documents:
    agent.add_memory(doc)
```

### 3. 问答

```python
# 单个问题
question = "什么是机器学习？"
answer = agent.QA(question)
print(f"Q: {question}\nA: {answer}")

# 批量问答
questions = [
    "什么是深度学习？",
    "NLP 的全称是什么？"
]
answers = agent.QA_batch(questions)

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}")
```

### 4. 查看记忆统计

```python
stats = agent.get_memory_stats()
print(f"记忆摘要数: {stats['num_abstracts']}")
print(f"页面数: {stats['num_pages']}")
print(f"检索器: {stats['retrievers']}")
```

### 5. 重置代理

```python
# 清空所有记忆和索引
agent.reset()
```

## 配置选项

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `client` | OpenAI | None | OpenAI 客户端实例 |
| `model_name` | str | "gpt4.1" | 模型名称 |
| `use_local_api` | bool | True | 是否使用本地 API |
| `max_research_iters` | int | 3 | ResearchAgent 最大迭代次数 |
| `retriever_types` | List[str] | ["bm25"] | 检索器类型列表 |
| `top_k` | int | 10 | 检索 top-k 结果 |
| `index_dir` | str | "./tmp/gam_indices" | 索引存储目录 |

### 检索器类型

- **`"index"`**: 基于索引的检索器（最快）
- **`"bm25"`**: BM25 关键词检索器（平衡）
- **`"dense"`**: 密集向量检索器（最准确，需要嵌入模型）

可以同时使用多个检索器：

```python
agent = GAMAgent(
    client=client,
    retriever_types=["bm25", "dense"],  # 组合使用
    # ...
)
```

## 在评估系统中使用

### 运行评估

```bash
# 使用 GAM agent 运行 Locomo 评估
python evaluate.py --task locomo --agent gam

# 使用 GAM agent 运行 LongMemEval 评估
python evaluate.py --task longmemeval --agent gam
```

### 自定义配置

在评估脚本中传递额外参数：

```python
# 在 evaluate.py 中修改 agent 初始化部分
agent = AGENT_CLASS[args.agent](
    client=client,
    model_name=args.model,
    use_local_api=True,
    max_research_iters=5,  # 增加迭代次数
    retriever_types=["bm25", "dense"],  # 使用多个检索器
    top_k=20  # 增加检索结果数
)
```

## 环境变量配置

在 `.env` 文件中配置：

```bash
# 远程 API 配置
X-CHJ-GWToken=your-token
X-CHJ-GW-SOURCE=your-source

# 本地 API 配置（用于 GAM）
LOCAL_API_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_API_KEY=EMPTY
GAM_MODEL_NAME=Qwen/Qwen3-8B
```

## 测试

运行测试脚本：

```bash
python test_gam_agent.py
```

测试内容包括：
1. 基本记忆添加和问答
2. 批量问答
3. 记忆统计
4. 代理重置
5. 评估格式兼容性

## 工作原理

### 记忆构建流程

1. **文本输入**: 通过 `add_memory()` 添加原始文本
2. **摘要生成**: MemoryAgent 自动生成记忆摘要
3. **页面存储**: 原始文本存储在 PageStore 中
4. **索引构建**: 检索器对页面内容建立索引

### 问答流程

1. **查询输入**: 通过 `QA()` 提交问题
2. **检索器初始化**: 首次问答时构建检索器
3. **相关内容检索**: 使用配置的检索器查找相关页面
4. **迭代推理**: ResearchAgent 进行多轮推理
5. **答案生成**: 综合检索结果生成最终答案

### Fallback 机制

当检索器不可用时，GAMAgent 会自动切换到 fallback 模式：
- 直接使用记忆摘要
- 通过 LLM 生成答案
- 保证基本功能可用

## 性能优化建议

1. **检索器选择**:
   - 快速测试: 使用 `["bm25"]`
   - 高准确度: 使用 `["bm25", "dense"]`
   - 最快速度: 使用 `["index"]`

2. **迭代次数**:
   - 简单问题: `max_research_iters=2`
   - 复杂问题: `max_research_iters=5`

3. **批量处理**:
   - 使用 `QA_batch()` 而非循环调用 `QA()`
   - 可以提高整体效率

4. **索引目录**:
   - 使用 SSD 存储索引文件
   - 定期清理不需要的索引

## 故障排查

### GAM 框架未找到

```
Warning: GAM framework not available
```

**解决方案**:
- 检查 GAM 框架是否安装在正确位置
- 确认路径: `../general-agentic-memory`
- 或修改 `gam_agent.py` 中的 `gam_path`

### 检索器创建失败

```
Warning: Failed to create BM25Retriever
```

**解决方案**:
- 检查是否安装了必要的依赖
- 确认索引目录有写权限
- 尝试使用不同的检索器类型

### API 调用失败

```
ERROR_API_CALL: Connection refused
```

**解决方案**:
- 检查本地 API 服务是否启动
- 验证 `base_url` 配置是否正确
- 尝试 `use_local_api=False` 使用远程 API

## 进阶用法

### 自定义检索器配置

```python
# 在 gam_agent.py 中修改检索器配置
dense_config = DenseRetrieverConfig(
    index_dir=dense_index_dir,
    model_name="BAAI/bge-large-zh-v1.5",  # 使用中文优化模型
    batch_size=32,
    max_length=512
)
```

### 异步支持

```python
from openai import AsyncOpenAI

# 创建异步客户端
async_client = AsyncOpenAI(...)

# 使用异步方法
answer = await agent.QA_async(question)
answers = await agent.QA_batch_async(questions)
```

## 相关资源

- GAM 框架文档: [GAM Docs](https://gam-docs.example.com)
- 项目 README: [README.md](../README.md)
- 示例代码: [examples/](../examples/)

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 GAMAgent！
