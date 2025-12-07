"""
RAGAgent - 简化的基于RAG的记忆代理
基于EmergenceAgent，移除事实提取步骤，直接使用检索到的对话生成答案
使用 OpenAI embedding API 进行向量检索
"""

import json
import textwrap
import os
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI

from .base_agent import BaseAgent, MODEL_NAME_MAP


class RAGAgent(BaseAgent):
    """基于语义检索的简化记忆代理，直接使用检索到的对话生成答案"""

    def __init__(self, client: OpenAI, top_k: int = 20, model_name: str = "gpt4.1"):
        super().__init__(client, model_name)

        self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.top_k = top_k

        self.corpus_embeddings = None
        self.all_turns = []
        self._is_async = self._check_if_async_client()

        # 初始化 embedding client
        self._init_embedding_client()

    def _check_if_async_client(self):
        """Check if client is AsyncOpenAI"""
        return hasattr(self.client, '__class__') and 'Async' in self.client.__class__.__name__

    def _init_embedding_client(self):
        """初始化 OpenAI embedding client"""
        embedding_endpoint = os.environ.get("EMBEDDING_SERVICE_ENDPOINT", "http://localhost:8000/v1/embeddings")
        base_url = embedding_endpoint.replace("embeddings", "").rstrip("/")

        if self._is_async:
            self.embedding_client = AsyncOpenAI(
                base_url=base_url,
                api_key="EMPTY"
            )
        else:
            self.embedding_client = OpenAI(
                base_url=base_url,
                api_key="EMPTY"
            )

        print(f"Initialized OpenAI embedding client with base_url: {base_url}")

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """使用 OpenAI API 编码文本（同步版本）"""
        if isinstance(texts, str):
            texts = [texts]

        response = self.embedding_client.embeddings.create(
            input=texts,
            model=self.embedding_model_name
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    async def _encode_texts_async(self, texts: List[str]) -> np.ndarray:
        """使用 OpenAI API 编码文本（异步版本）"""
        if isinstance(texts, str):
            texts = [texts]

        response = await self.embedding_client.embeddings.create(
            input=texts,
            model=self.embedding_model_name
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _cosine_similarity(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(corpus_norm, query_norm)
        return similarities

    def _semantic_search(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """语义搜索，返回 top_k 个最相关的结果"""
        similarities = self._cosine_similarity(query_embedding, corpus_embeddings)

        # Get top_k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        hits = [
            {'corpus_id': int(idx), 'score': float(similarities[idx])}
            for idx in top_indices
        ]

        return hits

    def add_memory_org(self, content: str) -> Dict[str, Any]:
        """添加记忆内容"""
        try:
            # 如果是JSON格式的haystack数据
            if content.strip().startswith('{'):
                try:
                    data = json.loads(content)
                    if 'haystack_sessions' in data and 'haystack_dates' in data:
                        return self._process_haystack(data['haystack_sessions'], data['haystack_dates'])
                except json.JSONDecodeError:
                    pass

            # 处理为简单文本
            self.all_turns.append(content)

            # 重新编码语料库
            if self.all_turns:
                self.corpus_embeddings = self._encode_texts(self.all_turns)

            return {"success": True, "total_turns": len(self.all_turns)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _process_haystack(self, haystack_sessions: List[List[Dict]], haystack_dates: List[str]) -> Dict[str, Any]:
        """处理haystack格式的记忆数据"""
        new_turns = []
        for session, date in zip(haystack_sessions, haystack_dates):
            for turn in session:
                if 'role' in turn and 'content' in turn:
                    new_turns.append(f"[{date}] {turn['role']}: {turn['content']}")

        self.all_turns.extend(new_turns)
        self.corpus_embeddings = self._encode_texts(self.all_turns)

        return {"success": True, "turns_added": len(new_turns), "total_turns": len(self.all_turns)}

    def QA(self, query: str) -> str:
        """回答问题 - 简化版，直接使用检索到的对话生成答案"""
        try:
            if self.corpus_embeddings is None or len(self.all_turns) == 0:
                return "I don't have any memory to answer questions from."

            # 步骤1: 检索相关记忆
            query_text = query[query.find(']')+1:].strip() if ']' in query else query
            query_embedding = self._encode_texts([query_text])[0]
            hits = self._semantic_search(query_embedding, self.corpus_embeddings, self.top_k)
            retrieved_turns = [self.all_turns[hit['corpus_id']] for hit in hits]

            # 步骤2: 直接生成答案（跳过事实提取步骤）
            answer = self._generate_answer(query, retrieved_turns)

            return answer.strip()

        except Exception as e:
            return f"Error: {str(e)}"

    def QA_batch(self, queries: List[str], batch_size: int = 32) -> List[str]:
        """批量回答问题"""
        if self._is_async:
            raise RuntimeError("Use QA_batch_async() for async client")
        return [self.QA(query) for query in queries]

    async def QA_async(self, query: str) -> str:
        """异步回答问题"""
        try:
            if self.corpus_embeddings is None or len(self.all_turns) == 0:
                return "I don't have any memory to answer questions from."

            # 步骤1: 检索相关记忆
            query_text = query[query.find(']')+1:].strip() if ']' in query else query
            query_embedding = await self._encode_texts_async([query_text])
            query_embedding = query_embedding[0]
            hits = self._semantic_search(query_embedding, self.corpus_embeddings, self.top_k)
            retrieved_turns = [self.all_turns[hit['corpus_id']] for hit in hits]

            # 直接生成答案
            answer = await self._generate_answer_async(query, retrieved_turns)

            return answer.strip()

        except Exception as e:
            return f"Error: {str(e)}"

    async def QA_batch_async(self, queries: List[str]) -> List[str]:
        """异步批量回答问题"""
        results = []
        for query in queries:
            result = await self.QA_async(query)
            results.append(result)
        return results

    def _generate_answer(self, question: str, retrieved_turns: List[str]) -> str:
        """直接使用检索到的对话生成最终答案（无事实提取步骤）"""
        answer_prompt = f"""
        You are a helpful assistant. Using the conversation turns below,
        answer the question as accurately and concisely as possible.

        Retrieved Conversation Turns:
        {json.dumps(retrieved_turns, indent=2)}

        Question: {question}
        Answer:
        """
        answer_prompt = textwrap.dedent(answer_prompt)

        try:
            answer = self.client.chat.completions.create(
                model=MODEL_NAME_MAP.get(self.model_name, self.model_name),
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=256
            ).choices[0].message.content
            return answer
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _parse_longmemeval_chunk(self, content: str) -> List[str]:
        """解析LongMemEval格式的chunk为individual turns（严格按照原始实现）"""
        turns = []
        lines = content.split('\n')

        # 找到日期和对话部分
        date_line = None
        conversation_started = False
        current_role = None
        current_content = ""

        for line in lines:
            line = line.strip()
            if line.startswith('DATE:'):
                # 提取日期
                date_part = line.split('DATE:')[1].strip()
                # 简化日期格式，提取主要部分
                if '(' in date_part:
                    date_line = date_part.split('(')[0].strip()
                else:
                    date_line = date_part
            elif line == 'CONVERSATION:':
                conversation_started = True
            elif conversation_started and line:
                # 解析对话turn - 检查是否是新的角色开始
                if line.startswith('User said,') or line.startswith('Assistant said,'):
                    # 如果之前有内容，先保存之前的turn
                    if current_role and current_content:
                        # 格式化为原始格式：[日期] 角色: 内容
                        if date_line:
                            turn = f"[{date_line}] {current_role}: {current_content}"
                        else:
                            turn = f"{current_role}: {current_content}"
                        turns.append(turn)

                    # 开始新的turn
                    if line.startswith('User said,'):
                        current_role = 'user'
                        content_part = line[len('User said,'):].strip()
                    else:  # Assistant said,
                        current_role = 'assistant'
                        content_part = line[len('Assistant said,'):].strip()

                    # 移除引号
                    if content_part.startswith('"') and content_part.endswith('"'):
                        content_part = content_part[1:-1]

                    current_content = content_part
                else:
                    # 这是同一个turn的继续内容，追加到current_content
                    if current_content:
                        current_content += "\n" + line

        # 处理最后一个turn
        if current_role and current_content:
            if date_line:
                turn = f"[{date_line}] {current_role}: {current_content}"
            else:
                turn = f"{current_role}: {current_content}"
            turns.append(turn)

        return turns

    def add_memory(self, content: str) -> Dict[str, Any]:
        """专门处理LongMemEval格式的add_memory方法"""
        try:
            # 解析LongMemEval格式的内容为turns
            if content.startswith('Below is a conversation between user and assistant.'):
                turns = self._parse_longmemeval_chunk(content)
                self.all_turns.extend(turns)
                turns_added = len(turns)
            else:
                # 如果不是LongMemEval格式，添加为单个turn
                self.all_turns.append(content)
                turns_added = 1

            # 重新编码语料库
            if self.all_turns:
                self.corpus_embeddings = self._encode_texts(self.all_turns)

            return {"success": True, "total_turns": len(self.all_turns), "turns_added": turns_added}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_answer_async(self, question: str, retrieved_turns: List[str]) -> str:
        """异步生成最终答案（无事实提取步骤）"""
        answer_prompt = f"""
        You are a helpful assistant. Using the conversation turns below,
        answer the question as accurately and concisely as possible.

        Retrieved Conversation Turns:
        {json.dumps(retrieved_turns, indent=2)}

        Question: {question}
        Answer:
        """
        answer_prompt = textwrap.dedent(answer_prompt)

        try:
            response = await self.client.chat.completions.create(
                model=MODEL_NAME_MAP.get(self.model_name, self.model_name),
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    async def add_memory_async(self, content: str) -> Dict[str, Any]:
        """异步处理LongMemEval格式的add_memory方法 - 增量编码优化"""
        try:
            # 记录当前turns数量，用于增量编码
            old_turns_count = len(self.all_turns)

            if content.startswith('Below is a conversation between user and assistant.'):
                turns = self._parse_longmemeval_chunk(content)
                self.all_turns.extend(turns)
                turns_added = len(turns)
            else:
                self.all_turns.append(content)
                turns_added = 1

            if self.all_turns:
                # 只编码新增的turns（增量编码）
                new_turns = self.all_turns[old_turns_count:]
                new_embeddings = await self._encode_texts_async(new_turns)

                # 拼接到现有embeddings
                if self.corpus_embeddings is None:
                    self.corpus_embeddings = new_embeddings
                else:
                    self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_embeddings])

            return {"success": True, "total_turns": len(self.all_turns), "turns_added": turns_added}

        except Exception as e:
            return {"success": False, "error": str(e)}
