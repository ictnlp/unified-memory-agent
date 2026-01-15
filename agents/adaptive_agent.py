"""
AdaptiveAgent - 自适应记忆代理
根据任务类型自动选择最优策略：
1. ICL分类任务 -> 全量样例或大量检索
2. 长对话记忆任务 -> RAG检索
3. 短上下文QA -> 直接Concat
"""

import json
import re
import textwrap
import os
import asyncio
import string
import numpy as np
from typing import List, Dict, Any, Literal, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from rank_bm25 import BM25Okapi

from .base_agent import BaseAgent, MODEL_NAME_MAP


class TaskType:
    """任务类型枚举"""
    ICL_CLASSIFICATION = "icl_classification"  # In-Context Learning分类任务
    LONG_MEMORY = "long_memory"                 # 长对话记忆任务
    SHORT_CONTEXT = "short_context"             # 短上下文QA任务


class AdaptiveAgent(BaseAgent):
    """
    自适应记忆代理，根据任务特征自动选择最优策略

    核心设计思想：
    1. 任务检测：通过启发式规则自动识别任务类型
    2. 策略适配：针对不同任务使用不同的检索/生成策略
    3. 动态调整：根据memory规模和问题特征动态调整参数
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str = "gpt4.1",
        # RAG检索参数
        top_k: int = 20,
        retrieval_method: str = "hybrid",  # "embedding", "bm25", "hybrid"
        rrf_k: int = 60,
        # 自适应参数
        auto_detect: bool = True,           # 是否自动检测任务类型
        icl_threshold: int = 50,            # ICL任务检测阈值（训练样例数量）
        long_memory_threshold: int = 100,   # 长记忆任务阈值（chunk数量）
        max_context_tokens: int = 12000,    # 最大context token数
    ):
        super().__init__(client, model_name)

        # RAG检索设置
        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.top_k = top_k
        self.retrieval_method = retrieval_method
        self.rrf_k = rrf_k

        # 自适应参数
        self.auto_detect = auto_detect
        self.icl_threshold = icl_threshold
        self.long_memory_threshold = long_memory_threshold
        self.max_context_tokens = max_context_tokens

        # 任务状态
        self.detected_task_type: Optional[str] = None
        self.icl_sample_count = 0  # ICL训练样例计数

        # Embedding检索相关
        self.corpus_embeddings = None
        self.embedding_client = None

        # BM25检索相关
        self.bm25 = None
        self.tokenized_corpus = []

        # 文档存储
        self.all_turns = []  # 所有记忆片段（可能是切分后的小块）
        self.original_chunks = []  # 原始chunk（不切分）

        # 初始化embedding client
        if self.retrieval_method in ["embedding", "hybrid"]:
            self._init_embedding_client()

    def _init_embedding_client(self):
        """初始化 OpenAI embedding client"""
        embedding_endpoint = os.environ.get("EMBEDDING_SERVICE_ENDPOINT")
        if not embedding_endpoint:
            print("[WARNING] EMBEDDING_SERVICE_ENDPOINT not set, embedding retrieval disabled")
            return

        base_url = embedding_endpoint.replace("embeddings", "").rstrip("/")
        self.embedding_client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )
        print(f"Initialized OpenAI embedding client with base_url: {base_url}")

    # ==================== 任务检测 ====================

    def _detect_task_type(self, content: str) -> str:
        """
        检测任务类型的启发式规则

        返回: TaskType中的一种
        """
        # 规则1: ICL分类任务特征
        icl_patterns = [
            r"These labeled training examples are for classification",
            r"Label:\s*\d+",
            r"Sentence:.*\nLabel:",
            r"What are the labels for the above sentence",
        ]

        icl_score = 0
        for pattern in icl_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                icl_score += 1

        # 检查是否包含大量标注样例
        label_matches = re.findall(r'\nLabel:\s*\d+', content)
        if len(label_matches) >= self.icl_threshold:
            return TaskType.ICL_CLASSIFICATION

        if icl_score >= 2:
            return TaskType.ICL_CLASSIFICATION

        # 规则2: 长对话记忆任务特征
        memory_patterns = [
            r"DATE:\s*\d{4}-\d{2}-\d{2}",  # 日期标记
            r"(user|assistant):",           # 对话角色
            r"Below is a conversation between",
        ]

        memory_score = sum(1 for p in memory_patterns if re.search(p, content, re.IGNORECASE))

        # 如果已经有很多chunks，很可能是长记忆任务
        if len(self.original_chunks) >= self.long_memory_threshold:
            return TaskType.LONG_MEMORY

        if memory_score >= 2:
            return TaskType.LONG_MEMORY

        # 默认：短上下文任务
        return TaskType.SHORT_CONTEXT

    def _update_task_detection(self):
        """基于当前所有memory更新任务类型检测"""
        if not self.auto_detect:
            return

        # 统计特征
        total_chunks = len(self.original_chunks)
        total_chars = sum(len(chunk) for chunk in self.original_chunks)

        # 检查ICL样例数量
        icl_count = 0
        for chunk in self.original_chunks[:10]:  # 只检查前10个chunk
            icl_count += len(re.findall(r'\nLabel:\s*\d+', chunk))

        self.icl_sample_count = icl_count

        # 决策逻辑
        if icl_count >= self.icl_threshold:
            self.detected_task_type = TaskType.ICL_CLASSIFICATION
        elif total_chunks >= self.long_memory_threshold:
            self.detected_task_type = TaskType.LONG_MEMORY
        elif total_chars > 50000:  # 超过50k字符
            self.detected_task_type = TaskType.LONG_MEMORY
        else:
            self.detected_task_type = TaskType.SHORT_CONTEXT

        print(f"[TaskDetection] Type: {self.detected_task_type}, Chunks: {total_chunks}, ICL samples: {icl_count}")

    # ==================== 检索方法 ====================

    async def _encode_texts_async(self, texts: List[str]) -> np.ndarray:
        """使用 OpenAI API 编码文本（异步版本）"""
        if isinstance(texts, str):
            texts = [texts]

        if not self.embedding_client:
            raise RuntimeError("Embedding client not initialized")

        response = await self.embedding_client.embeddings.create(
            input=texts,
            model=self.embedding_model_name
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _cosine_similarity(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        similarities = np.dot(corpus_norm, query_norm)
        return similarities

    def _semantic_search(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """语义搜索"""
        similarities = self._cosine_similarity(query_embedding, corpus_embeddings)
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
        """使用Reciprocal Rank Fusion融合BM25和embedding检索结果"""
        if not self.all_turns:
            return []

        retrieval_k = min(top_k * 2, len(self.all_turns))

        # BM25检索
        bm25_hits = self._bm25_search(query, retrieval_k)

        # Embedding检索
        if self.corpus_embeddings is not None:
            embedding_hits = self._semantic_search(query_embedding, self.corpus_embeddings, retrieval_k)
        else:
            embedding_hits = []

        # RRF融合
        rrf_scores = {}
        for rank, hit in enumerate(bm25_hits, start=1):
            corpus_id = hit['corpus_id']
            rrf_scores[corpus_id] = rrf_scores.get(corpus_id, 0) + 1.0 / (self.rrf_k + rank)

        for rank, hit in enumerate(embedding_hits, start=1):
            corpus_id = hit['corpus_id']
            rrf_scores[corpus_id] = rrf_scores.get(corpus_id, 0) + 1.0 / (self.rrf_k + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(sorted_results))

        hits = [
            {'corpus_id': corpus_id, 'score': score}
            for corpus_id, score in sorted_results[:top_k]
        ]
        return hits

    # ==================== 自适应检索策略 ====================

    async def _adaptive_retrieve(self, query: str) -> List[str]:
        """
        自适应检索策略：根据任务类型选择最优方法

        返回: 检索到的文本片段列表
        """
        task_type = self.detected_task_type or TaskType.SHORT_CONTEXT

        # 策略1: ICL分类任务 - 使用大量或全量样例
        if task_type == TaskType.ICL_CLASSIFICATION:
            print(f"[AdaptiveRetrieve] ICL task detected, using large sample strategy")

            # 如果原始chunks不多，直接使用全量
            if len(self.original_chunks) <= 200:
                return self.original_chunks

            # 否则，检索更多样例（100-200个chunk）
            retrieve_k = min(200, len(self.all_turns))

            # 优先使用BM25，因为ICL样例的匹配更依赖关键词
            if self.bm25 is not None:
                hits = self._bm25_search(query, retrieve_k)
            else:
                # Fallback to embedding
                query_embedding = await self._encode_texts_async([query])
                hits = self._semantic_search(query_embedding[0], self.corpus_embeddings, retrieve_k)

            retrieved = [self.all_turns[hit['corpus_id']] for hit in hits]
            return retrieved

        # 策略2: 长记忆任务 - 使用混合检索
        elif task_type == TaskType.LONG_MEMORY:
            print(f"[AdaptiveRetrieve] Long memory task, using hybrid retrieval (k={self.top_k})")

            query_text = query[query.find(']')+1:].strip() if ']' in query else query

            if self.retrieval_method == "hybrid" and self.corpus_embeddings is not None:
                query_embedding = await self._encode_texts_async([query_text])
                hits = self._rrf_search(query_text, query_embedding[0], self.top_k)
            elif self.retrieval_method == "bm25":
                hits = self._bm25_search(query_text, self.top_k)
            else:
                query_embedding = await self._encode_texts_async([query_text])
                hits = self._semantic_search(query_embedding[0], self.corpus_embeddings, self.top_k)

            retrieved = [self.all_turns[hit['corpus_id']] for hit in hits]
            return retrieved

        # 策略3: 短上下文任务 - 直接使用全量
        else:
            print(f"[AdaptiveRetrieve] Short context task, using all content")

            # 如果内容不多，直接返回全部
            if len(self.original_chunks) <= 50:
                return self.original_chunks

            # 否则使用检索但提高k值
            retrieve_k = min(50, len(self.all_turns))
            query_text = query[query.find(']')+1:].strip() if ']' in query else query

            if self.corpus_embeddings is not None:
                query_embedding = await self._encode_texts_async([query_text])
                hits = self._semantic_search(query_embedding[0], self.corpus_embeddings, retrieve_k)
                retrieved = [self.all_turns[hit['corpus_id']] for hit in hits]
            else:
                retrieved = self.all_turns[-retrieve_k:]  # 使用最近的k个

            return retrieved

    # ==================== Memory管理 ====================

    def _parse_longmemeval_chunk(self, content: str) -> List[str]:
        """解析LongMemEval格式的chunk为individual turns"""
        turns = []
        lines = content.split('\n')

        date_line = None
        conversation_started = False
        current_role = None
        current_content = ""

        for line in lines:
            line = line.strip()
            if line.startswith('DATE:'):
                date_part = line.split('DATE:')[1].strip()
                if '(' in date_part:
                    date_line = date_part.split('(')[0].strip()
                else:
                    date_line = date_part
            elif line == 'CONVERSATION:':
                conversation_started = True
            elif conversation_started and line:
                if line.startswith('User said,') or line.startswith('Assistant said,'):
                    if current_role and current_content:
                        if date_line:
                            turn = f"[{date_line}] {current_role}: {current_content}"
                        else:
                            turn = f"{current_role}: {current_content}"
                        turns.append(turn)

                    if line.startswith('User said,'):
                        current_role = 'user'
                        content_part = line[len('User said,'):].strip()
                    else:
                        current_role = 'assistant'
                        content_part = line[len('Assistant said,'):].strip()

                    if content_part.startswith('"') and content_part.endswith('"'):
                        content_part = content_part[1:-1]

                    current_content = content_part
                else:
                    if current_content:
                        current_content += "\n" + line

        if current_role and current_content:
            if date_line:
                turn = f"[{date_line}] {current_role}: {current_content}"
            else:
                turn = f"{current_role}: {current_content}"
            turns.append(turn)

        return turns

    def _should_chunk_split(self, content: str, task_type: str) -> bool:
        """决定是否需要切分chunk"""
        # ICL任务：不切分，保持训练样例完整性
        if task_type == TaskType.ICL_CLASSIFICATION:
            return False

        # 短内容：不需要切分
        if len(content) < 2000:
            return False

        # 长对话：可以切分成小块以提高检索精度
        if task_type == TaskType.LONG_MEMORY:
            return True

        return False

    async def add_memory_async(self, content: str) -> Dict[str, Any]:
        """
        异步添加memory，使用自适应策略
        """
        try:
            # 保存原始chunk
            self.original_chunks.append(content)

            # 检测任务类型
            detected_type = self._detect_task_type(content)

            # 决定是否切分
            should_split = self._should_chunk_split(content, detected_type)

            # 处理content
            if content.startswith('Below is a conversation between user and assistant.'):
                # LongMemEval格式：解析为turns
                turns = self._parse_longmemeval_chunk(content)
                self.all_turns.extend(turns)
                turns_added = len(turns)
            elif should_split:
                # 切分成小块
                from agents.rag_agent import get_chunks
                subchunks = get_chunks(content, 1000)
                self.all_turns.extend(subchunks)
                turns_added = len(subchunks)
            else:
                # 保持完整
                self.all_turns.append(content)
                turns_added = 1

            # 更新任务检测
            self._update_task_detection()

            # 构建索引
            if self.all_turns:
                if self.retrieval_method in ["hybrid", "bm25"]:
                    # 构建BM25索引
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

                if self.retrieval_method in ["hybrid", "embedding"] and self.embedding_client:
                    # 增量构建embedding索引
                    old_count = len(self.corpus_embeddings) if self.corpus_embeddings is not None else 0
                    new_turns = self.all_turns[old_count:]
                    if new_turns:
                        new_embeddings = await self._encode_texts_async(new_turns)
                        if self.corpus_embeddings is None:
                            self.corpus_embeddings = new_embeddings
                        else:
                            self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_embeddings])

            return {
                "success": True,
                "total_turns": len(self.all_turns),
                "turns_added": turns_added,
                "detected_task_type": self.detected_task_type
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== Question Answering ====================

    async def _generate_answer_async(self, question: str, retrieved_turns: List[str]) -> str:
        """生成答案"""
        retrieved_turns_str = json.dumps(retrieved_turns, indent=2, ensure_ascii=False)

        question_part = f"\n\nQuestion: {question}\nAnswer:"
        template_part = textwrap.dedent("""
        You are a helpful assistant. Using the conversation turns below,
        answer the question as accurately and concisely as possible.

        Retrieved Conversation Turns:
        """)

        # 动态计算max context
        max_context_chars = self.max_context_tokens * 4 - len(template_part) - len(question_part) - 500

        if len(retrieved_turns_str) > max_context_chars:
            print(f"[WARNING] Retrieved turns too long ({len(retrieved_turns_str)} chars), truncating to {max_context_chars} chars")
            retrieved_turns_str = retrieved_turns_str[:max_context_chars] + "\n...[Content truncated]..."

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

    async def QA_async(self, query: str) -> str:
        """异步回答问题，使用自适应检索策略"""
        try:
            if len(self.all_turns) == 0:
                return "I don't have any memory to answer questions from."

            # 自适应检索
            retrieved_turns = await self._adaptive_retrieve(query)

            # 生成答案
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
