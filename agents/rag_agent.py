"""
RAGAgent - 简化的基于RAG的记忆代理
基于EmergenceAgent，移除事实提取步骤，直接使用检索到的对话生成答案
支持两种检索方式：embedding (语义检索) 和 bm25 (关键词检索)
"""

import json
import textwrap
import os
import asyncio
import string
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from rank_bm25 import BM25Okapi

from .base_agent import BaseAgent, MODEL_NAME_MAP

def get_chunks(context_text, chunk_size):
    separator = "\n"
    small_chunks = context_text.split(separator)
    chunks = []
    to_append = ""
    for chunk in small_chunks:
        if len(to_append) + len(chunk) + len(separator) <= chunk_size:
            if to_append:
                to_append += separator + chunk
            else:
                to_append = chunk
        else:
            if to_append:
                chunks.append(to_append)
            to_append = chunk
    if to_append:
        chunks.append(to_append)
    return chunks

class RAGAgent(BaseAgent):
    """基于检索的简化记忆代理，支持embedding、bm25和混合检索(RRF)三种方式"""

    def __init__(self, client: OpenAI, top_k: int = 20, model_name: str = "gpt4.1", retrieval_method: str = "hybrid", rrf_k: int = 60):
        super().__init__(client, model_name)

        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.top_k = top_k
        self.retrieval_method = retrieval_method  # "embedding", "bm25", or "hybrid"
        self.rrf_k = rrf_k  # RRF常数，通常设置为60

        # Embedding检索相关
        self.corpus_embeddings = None

        # BM25检索相关
        self.bm25 = None
        self.tokenized_corpus = []

        # 共享的文档存储
        self.all_turns = []

        # 如果使用embedding或hybrid，需要初始化embedding client
        if self.retrieval_method in ["embedding", "hybrid"]:
            self._init_embedding_client()

    def _init_embedding_client(self):
        """初始化 OpenAI embedding client"""
        embedding_endpoint = os.environ["EMBEDDING_SERVICE_ENDPOINT"]
        base_url = embedding_endpoint.replace("embeddings", "").rstrip("/")

        self.embedding_client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )

        print(f"Initialized OpenAI embedding client with base_url: {base_url}")

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

    def _tokenize(self, text: str) -> List[str]:
        """分词函数，用于BM25检索"""
        translator = str.maketrans("", "", string.punctuation)
        cleaned = text.lower().translate(translator)
        return [token for token in cleaned.split() if token]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """使用BM25进行检索"""
        if self.bm25 is None or not self.all_turns:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = [
            {'corpus_id': int(idx), 'score': float(scores[idx])}
            for idx in top_indices
        ]

        return hits

    def _rrf_search(self, query: str, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        使用Reciprocal Rank Fusion (RRF)融合BM25和embedding检索结果

        RRF公式: RRF_score(d) = Σ 1/(k + rank_i(d))
        其中 k 是常数(默认60)，rank_i(d) 是文档d在第i个排序列表中的排名(从1开始)
        """
        if not self.all_turns:
            return []

        # 获取两种检索方法的结果（取更多结果用于融合）
        retrieval_k = min(top_k * 2, len(self.all_turns))

        # BM25检索
        bm25_hits = self._bm25_search(query, retrieval_k)

        # Embedding检索
        if self.corpus_embeddings is not None:
            embedding_hits = self._semantic_search(query_embedding, self.corpus_embeddings, retrieval_k)
        else:
            embedding_hits = []

        # 使用RRF融合
        rrf_scores = {}

        # 计算BM25的RRF分数
        for rank, hit in enumerate(bm25_hits, start=1):
            corpus_id = hit['corpus_id']
            rrf_scores[corpus_id] = rrf_scores.get(corpus_id, 0) + 1.0 / (self.rrf_k + rank)

        # 计算Embedding的RRF分数
        for rank, hit in enumerate(embedding_hits, start=1):
            corpus_id = hit['corpus_id']
            rrf_scores[corpus_id] = rrf_scores.get(corpus_id, 0) + 1.0 / (self.rrf_k + rank)

        # 按RRF分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回top_k结果
        top_k = min(top_k, len(sorted_results))
        hits = [
            {'corpus_id': corpus_id, 'score': score}
            for corpus_id, score in sorted_results[:top_k]
        ]

        return hits

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

    async def QA_async(self, query: str) -> str:
        """异步回答问题"""
        try:
            if len(self.all_turns) == 0:
                return "I don't have any memory to answer questions from."

            # 步骤1: 检索相关记忆
            query_text = query[query.find(']')+1:].strip() if ']' in query else query

            if self.retrieval_method == "bm25":
                # 使用BM25检索
                hits = self._bm25_search(query_text, self.top_k)
            elif self.retrieval_method == "hybrid":
                # 使用混合检索(RRF)
                query_embedding = await self._encode_texts_async([query_text])
                query_embedding = query_embedding[0]
                hits = self._rrf_search(query_text, query_embedding, self.top_k)
            else:
                # 使用embedding检索
                if self.corpus_embeddings is None:
                    return "I don't have any memory to answer questions from."
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
        semaphore = asyncio.Semaphore(10)
        
        async def qa_with_semaphore(query):
            async with semaphore:
                return await self.QA_async(query)
        
        tasks = [qa_with_semaphore(query) for query in queries]
        results = await tqdm_asyncio.gather(*tasks, desc="Answering questions")
        return results

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

    async def _generate_answer_async(self, question: str, retrieved_turns: List[str]) -> str:
        retrieved_turns_str = json.dumps(retrieved_turns, indent=2)
        
        question_part = f"\n\nQuestion: {question}\nAnswer:"
        template_part = textwrap.dedent("""
        You are a helpful assistant. Using the conversation turns below,
        answer the question as accurately and concisely as possible.

        Retrieved Conversation Turns:
        """)
        
        max_context_chars = 49152 - 1024 * 3 - len(template_part) - len(question_part) - 100
        
        if len(retrieved_turns_str) > max_context_chars:
            print(f"[WARNING] Retrieved turns too long ({len(retrieved_turns_str)} chars), truncating to {max_context_chars} chars (keeping most relevant)")
            retrieved_turns_str = retrieved_turns_str[:max_context_chars] + "\n...[Later turns truncated]..."
        
        answer_prompt = template_part + retrieved_turns_str + question_part

        try:
            response = await self.client.chat.completions.create(
                model=MODEL_NAME_MAP.get(self.model_name, self.model_name),
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=1024
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
                # subchunks = get_chunks(content, 1000)
                # self.all_turns.extend(subchunks)
                # turns_added = len(subchunks)

            if self.all_turns:
                if self.retrieval_method == "hybrid":
                    # 混合检索：需要同时构建BM25和embedding索引
                    # 1. 构建BM25索引
                    self.tokenized_corpus = [self._tokenize(turn) for turn in self.all_turns]
                    valid_docs = []
                    valid_tokens = []
                    for turn, tokens in zip(self.all_turns, self.tokenized_corpus):
                        if tokens:
                            valid_docs.append(turn)
                            valid_tokens.append(tokens)
                    self.all_turns = valid_docs
                    self.tokenized_corpus = valid_tokens
                    self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

                    # 2. 增量构建embedding索引
                    new_turns = self.all_turns[old_turns_count:]
                    if new_turns:
                        new_embeddings = await self._encode_texts_async(new_turns)
                        if self.corpus_embeddings is None:
                            self.corpus_embeddings = new_embeddings
                        else:
                            self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_embeddings])

                elif self.retrieval_method == "bm25":
                    # 仅构建BM25索引
                    self.tokenized_corpus = [self._tokenize(turn) for turn in self.all_turns]
                    valid_docs = []
                    valid_tokens = []
                    for turn, tokens in zip(self.all_turns, self.tokenized_corpus):
                        if tokens:
                            valid_docs.append(turn)
                            valid_tokens.append(tokens)
                    self.all_turns = valid_docs
                    self.tokenized_corpus = valid_tokens
                    self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

                else:
                    # 仅构建Embedding索引（增量）
                    new_turns = self.all_turns[old_turns_count:]
                    new_embeddings = await self._encode_texts_async(new_turns)
                    if self.corpus_embeddings is None:
                        self.corpus_embeddings = new_embeddings
                    else:
                        self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_embeddings])

            return {"success": True, "total_turns": len(self.all_turns), "turns_added": turns_added}

        except Exception as e:
            return {"success": False, "error": str(e)}
