# GAMAgent 实现总结

## 修改内容

### 1. 简化的异步实现 (`agents/gam_agent.py`)

#### 核心特性
- ✅ **仅保留异步方法**: `add_memory_async()` 和 `QA_batch_async()`
- ✅ **移除同步方法**: 删除了 `add_memory()`, `QA()`, `QA_batch()` 等同步方法
- ✅ **移除 use_local_api 判断**: 统一使用传入的 client，不再区分本地/远程 API
- ✅ **使用传入的 model_name**: 通过 `MODEL_NAME_MAP` 正确映射模型名称

#### AsyncOpenAIGeneratorWrapper 类

创建了一个包装类，将 `AsyncOpenAI` client 包装成符合 GAM `OpenAIGenerator` 接口的形式：

```python
class AsyncOpenAIGeneratorWrapper:
    """包装 AsyncOpenAI client，暴露和 OpenAIGenerator 一样的接口"""

    async def generate_single(self, prompt=None, messages=None, schema=None, extra_params=None):
        """生成单个响应（异步）"""
        # 返回: {"text": str, "json": dict|None, "response": dict}

    async def generate_batch(self, prompts=None, messages_list=None, schema=None, extra_params=None):
        """批量生成响应（异步并发）"""
        # 返回: [{"text": str, "json": dict|None, "response": dict}, ...]
```

**核心功能**:
- 将 `AsyncOpenAI` 的调用方式转换为 GAM 期望的格式
- 支持 prompt 或 messages 两种输入方式
- 支持 JSON schema 结构化输出
- 内置重试逻辑（最多重试 3 次）
- 异步并发批量处理

#### GAMAgent 类

```python
class GAMAgent(BaseAgent):
    """基于 GAM 框架的记忆代理（异步版本）"""

    def __init__(self, client: AsyncOpenAI, model_name: str, ...):
        # 初始化 GAM 组件

    async def add_memory_async(self, chunk: str):
        # 使用 asyncio.to_thread 包装同步的 memorize 方法

    async def QA_batch_async(self, query_list: List[str], batch_size: int = 32):
        # 并发调用 ResearchAgent.research()

    def reset(self):
        # 重置状态和清理索引
```

**工作流程**:

1. **初始化阶段**:
   - 创建 `AsyncOpenAIGeneratorWrapper` 包装传入的 client
   - 初始化 `MemoryAgent` (用于记忆构建)
   - 延迟初始化 `ResearchAgent` (在首次问答时构建)

2. **添加记忆**:
   - 调用 `add_memory_async(chunk)`
   - 使用 `asyncio.to_thread` 包装同步的 `memory_agent.memorize()`
   - MemoryAgent 自动生成记忆摘要并存储

3. **问答处理**:
   - 首次调用时构建检索器（BM25/Index/Dense）
   - 使用 `asyncio.gather` 并发处理多个问题
   - 每个问题通过 `research_agent.research()` 进行多轮迭代推理
   - Fallback 机制：检索器不可用时使用记忆摘要直接生成答案

### 2. 异步测试脚本 (`test_gam_agent.py`)

- 移除所有同步测试
- 只保留异步测试方法
- 使用 `asyncio.run()` 运行测试

### 3. 配置注册 (`config.py`)

```python
'gam': ('agents.gam_agent', 'GAMAgent')
```

## 使用方法

### 在 evaluate_async.py 中使用

```bash
python evaluate_async.py --task locomo --agent gam --model qwen3-8b --concurrency 32
```

### 代码示例

```python
from openai import AsyncOpenAI
from config import AGENT_CLASS

# 创建客户端
client = AsyncOpenAI(base_url="...", api_key="...")

# 初始化 agent
GAMAgent = AGENT_CLASS['gam']
agent = GAMAgent(
    client=client,
    model_name="qwen3-8b",
    max_research_iters=3,
    retriever_types=["bm25"]
)

# 添加记忆
await agent.add_memory_async("Some memory chunk...")

# 批量问答
questions = ["Q1?", "Q2?", "Q3?"]
answers = await agent.QA_batch_async(questions)
```

## 技术细节

### AsyncOpenAI Client 包装

GAM 的 `OpenAIGenerator` 期望：
- `generate_single()` 返回 `{"text": str, "json": dict, "response": dict}`
- `generate_batch()` 返回列表

我们的 wrapper 实现：
- 使用 `AsyncOpenAI.chat.completions.create()`
- 格式化返回值匹配 GAM 期望
- 使用 `asyncio.gather()` 实现批量并发

### 同步到异步转换

GAM 的某些方法可能是同步的（如 `memorize()`, `research()`），使用 `asyncio.to_thread()` 包装：

```python
await asyncio.to_thread(self.memory_agent.memorize, chunk)
await asyncio.to_thread(self.research_agent.research, query)
```

### 检索器延迟初始化

- 首次调用 `QA_batch_async()` 时构建检索器
- 避免在初始化阶段占用过多时间
- 支持 Index, BM25, Dense 三种检索器

## 优势

1. **完全异步**: 与 evaluate_async.py 完美集成
2. **高并发**: 利用 asyncio 实现问题并发处理
3. **清晰简洁**: 只保留必要的异步方法
4. **正确的模型映射**: 使用传入的 model_name 而非硬编码
5. **灵活的检索**: 支持多种检索器组合
6. **Fallback 机制**: 检索器失败时仍能工作

## 注意事项

1. 需要确保 GAM 框架已安装在 `../general-agentic-memory`
2. 首次问答时会构建检索器索引，可能需要一些时间
3. 如果 GAM 不可用，初始化时会抛出 RuntimeError
4. 检索器构建失败会自动降级到 fallback 模式

## 测试

运行测试：
```bash
python test_gam_agent.py
```

测试内容：
- 异步记忆添加
- 异步批量问答
- 代理重置
- 评估格式兼容性
