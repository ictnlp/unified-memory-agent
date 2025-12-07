"""
GAMAgent - 基于 GAM (General Agentic Memory) 框架的记忆代理
集成 MemoryAgent 和 ResearchAgent 实现高级记忆构建和问答功能
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from .base_agent import BaseAgent, MODEL_NAME_MAP

# Import GAM components (path managed by agents/__init__.py)
try:
    from gam import (
        MemoryAgent,
        ResearchAgent,
        InMemoryMemoryStore,
        InMemoryPageStore,
        IndexRetriever,
        IndexRetrieverConfig,
        BM25Retriever,
        BM25RetrieverConfig,
        DenseRetriever,
        DenseRetrieverConfig,
    )
    GAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GAM framework not available: {e}")
    print("GAMAgent will not be functional. Please install GAM framework.")
    GAM_AVAILABLE = False


class AsyncOpenAIGeneratorWrapper:
    """
    包装 AsyncOpenAI client，暴露和 OpenAIGenerator 一样的接口
    让 GAM 的 MemoryAgent 和 ResearchAgent 可以使用
    """

    def __init__(self, client: AsyncOpenAI, model_name: str, temperature: float = 0.3, max_tokens: int = 512):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = None

    async def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成单个响应（异步）
        返回: {"text": str, "json": dict|None, "response": dict}
        """
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")

        # 构造 messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        # 构造 response_format
        response_format = None
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "auto_schema",
                    "schema": schema,
                    "strict": True
                }
            }

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        # 重试逻辑
        times = 0
        while True:
            try:
                resp = await self.client.chat.completions.create(**params)
                break
            except Exception as e:
                print(f"API call failed: {e}, retry times: {times}")
                times += 1
                if times > 3:
                    raise e
                await asyncio.sleep(5)

        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""

        out: Dict[str, Any] = {"text": text, "json": None, "response": resp.model_dump()}

        if schema is not None:
            try:
                out["json"] = json.loads(text)
            except Exception:
                out["json"] = None

        return out

    async def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应（异步并发）
        返回格式: [{"text": str, "json": dict|None, "response": dict}, ...]
        """
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # 异步并发调用
        import asyncio
        tasks = [
            self.generate_single(messages=messages, schema=schema, extra_params=extra_params)
            for messages in messages_list
        ]
        results = await asyncio.gather(*tasks)
        return results


class GAMAgent(BaseAgent):
    """
    基于 GAM 框架的记忆代理（异步版本）
    使用 MemoryAgent 进行记忆构建，使用 ResearchAgent 进行问答
    """

    def __init__(
        self,
        client: AsyncOpenAI = None,
        model_name: str = "gpt4.1",
        max_research_iters: int = 3,
        retriever_types: List[str] = ["bm25"],  # 可选: "index", "bm25", "dense"
        top_k: int = 10,
        index_dir: str = "./tmp/gam_indices"
    ):
        """
        初始化 GAM Agent

        Args:
            client: AsyncOpenAI client
            model_name: 模型名称
            max_research_iters: ResearchAgent最大迭代次数
            retriever_types: 使用的检索器类型列表
            top_k: 检索top-k结果
            index_dir: 索引存储目录
        """
        super().__init__(client, model_name)

        if not GAM_AVAILABLE:
            raise RuntimeError("GAM framework is not available. Cannot initialize GAMAgent.")

        self.max_research_iters = max_research_iters
        self.retriever_types = retriever_types
        self.top_k = top_k
        self.index_dir = index_dir

        # 初始化 GAM 组件
        self._init_gam_components()

    def _init_gam_components(self):
        """初始化 GAM 框架的各个组件"""
        # 1. 映射模型名称
        gam_model_name = MODEL_NAME_MAP.get(self.model_name, self.model_name)

        # 2. 创建 Generator wrapper
        self.generator = AsyncOpenAIGeneratorWrapper(
            client=self.client,
            model_name=gam_model_name,
            temperature=0.3,
            max_tokens=512
        )

        # 3. 创建存储
        self.memory_store = InMemoryMemoryStore()
        self.page_store = InMemoryPageStore()

        # 4. 创建 MemoryAgent
        self.memory_agent = MemoryAgent(
            generator=self.generator,
            memory_store=self.memory_store,
            page_store=self.page_store
        )

        # 5. 检索器和 ResearchAgent (延迟初始化)
        self.retrievers = None
        self.research_agent = None

        print(f"GAMAgent initialized with model: {gam_model_name}, retriever_types: {self.retriever_types}")

    async def _build_retrievers(self):
        """构建检索器 (在第一次问答时调用)"""
        if self.retrievers is not None:
            return

        self.retrievers = {}
        os.makedirs(self.index_dir, exist_ok=True)

        # 构建各种检索器
        if "index" in self.retriever_types:
            try:
                import shutil
                page_index_dir = os.path.join(self.index_dir, "page_index")
                if os.path.exists(page_index_dir):
                    shutil.rmtree(page_index_dir)

                index_config = IndexRetrieverConfig(index_dir=page_index_dir)
                index_retriever = IndexRetriever(index_config.__dict__)
                index_retriever.build(self.page_store)
                self.retrievers["page_index"] = index_retriever
                print("✓ IndexRetriever created")
            except Exception as e:
                print(f"Warning: Failed to create IndexRetriever: {e}")

        if "bm25" in self.retriever_types:
            try:
                import shutil
                bm25_index_dir = os.path.join(self.index_dir, "bm25_index")
                if os.path.exists(bm25_index_dir):
                    shutil.rmtree(bm25_index_dir)

                bm25_config = BM25RetrieverConfig(
                    index_dir=bm25_index_dir,
                    threads=1
                )
                bm25_retriever = BM25Retriever(bm25_config.__dict__)
                bm25_retriever.build(self.page_store)
                self.retrievers["keyword"] = bm25_retriever
                print("✓ BM25Retriever created")
            except Exception as e:
                print(f"Warning: Failed to create BM25Retriever: {e}")

        if "dense" in self.retriever_types:
            try:
                import shutil
                dense_index_dir = os.path.join(self.index_dir, "dense_index")
                if os.path.exists(dense_index_dir):
                    shutil.rmtree(dense_index_dir)

                dense_config = DenseRetrieverConfig(
                    index_dir=dense_index_dir,
                    model_name="BAAI/bge-m3"
                )
                dense_retriever = DenseRetriever(dense_config.__dict__)
                dense_retriever.build(self.page_store)
                self.retrievers["vector"] = dense_retriever
                print("✓ DenseRetriever created")
            except Exception as e:
                print(f"Warning: Failed to create DenseRetriever: {e}")

        # 6. 创建 ResearchAgent
        if self.retrievers:
            research_agent_kwargs = {
                "page_store": self.page_store,
                "memory_store": self.memory_store,
                "retrievers": self.retrievers,
                "generator": self.generator,
                "max_iters": self.max_research_iters
            }
            self.research_agent = ResearchAgent(**research_agent_kwargs)
            print(f"✓ ResearchAgent created with {len(self.retrievers)} retrievers")
        else:
            print("Warning: No retrievers available. Will use fallback QA method.")

    async def add_memory_async(self, chunk: str):
        """
        添加记忆块（异步）
        使用 MemoryAgent.memorize() 方法
        """
        try:
            # GAM 的 memorize 可能是同步的，需要用 to_thread 包装
            import asyncio
            await asyncio.to_thread(self.memory_agent.memorize, chunk)
        except Exception as e:
            print(f"Error adding memory: {e}")

    async def QA_batch_async(self, query_list: List[str]) -> List[str]:
        """
        异步批量问答
        使用 ResearchAgent.research() 方法
        """
        try:
            # 确保检索器已构建
            if self.research_agent is None:
                await self._build_retrievers()

            # 如果没有可用的研究代理，使用简单的fallback
            if self.research_agent is None:
                return await self._fallback_qa_batch(query_list)

            # 使用 ResearchAgent 逐个进行问答（因为 research 方法可能不支持批量）
            import asyncio

            async def research_single(query: str) -> str:
                """单个问题的研究"""
                try:
                    # GAM 的 research 可能是同步的，需要用 to_thread 包装
                    research_result = await asyncio.to_thread(self.research_agent.research, query)
                    answer = research_result.integrated_memory
                    return answer.strip() if answer else "I don't have enough information to answer this question."
                except Exception as e:
                    return self._handle_api_error(e, query)

            # 并发处理所有问题
            results = await asyncio.gather(*[research_single(q) for q in query_list])
            return results

        except Exception as e:
            print(f"Error in QA_batch_async: {e}")
            return [self._handle_api_error(e, q) for q in query_list]

    async def _fallback_qa_batch(self, query_list: List[str]) -> List[str]:
        """
        当检索器不可用时的fallback批量问答方法
        直接使用记忆摘要生成答案
        """
        try:
            memory_state = self.memory_store.load()

            if not memory_state or not memory_state.abstracts:
                return ["I don't have any memory to answer questions from."] * len(query_list)

            # 获取所有记忆摘要
            abstracts = memory_state.abstracts
            memory_context = "\n".join([f"Memory {i+1}: {abstract}"
                                       for i, abstract in enumerate(abstracts)])

            # 批量生成答案
            results = []
            for query in query_list:
                prompt = f"""Based on the following memory abstracts, answer the question concisely.

Memory Abstracts:
{memory_context}

Question: {query}
Answer:"""

                try:
                    response = await self.client.chat.completions.create(
                        model=MODEL_NAME_MAP.get(self.model_name, self.model_name),
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content
                    results.append(answer.strip())
                except Exception as e:
                    results.append(self._handle_api_error(e, query))

            return results

        except Exception as e:
            return [self._handle_api_error(e, q) for q in query_list]

    def reset(self) -> None:
        """
        重置代理状态
        清空所有记忆和索引
        """
        # 重新初始化 GAM 组件
        self._init_gam_components()

        # 清理索引目录
        if os.path.exists(self.index_dir):
            import shutil
            shutil.rmtree(self.index_dir)
            os.makedirs(self.index_dir, exist_ok=True)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        """
        try:
            memory_state = self.memory_store.load()
            page_ids = self.page_store.list()

            return {
                "num_abstracts": len(memory_state.abstracts) if memory_state else 0,
                "num_pages": len(page_ids),
                "retrievers": list(self.retrievers.keys()) if self.retrievers else []
            }
        except Exception as e:
            return {"error": str(e)}
