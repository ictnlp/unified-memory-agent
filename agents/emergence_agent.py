"""
EmergenceAgent - 基于RAG的简化记忆代理
基于emergence_simple_fast/main.py的核心实现
"""

import json
import textwrap
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

from .base_agent import BaseAgent, MODEL_NAME_MAP


class EmergenceAgent(BaseAgent):
    """基于语义检索和事实提取的简化记忆代理"""
    
    # 类级别共享的embedding模型（所有实例共享）
    _shared_retrieval_model = None
    _model_lock = None
    _encode_lock = None  # 用于保护 encode 操作的线程安全
    
    def __init__(self, client: OpenAI, top_k: int = 42, model_name: str = "gpt4.1"):
        super().__init__(client, model_name)
        
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.top_k = top_k
        
        self.corpus_embeddings = None
        self.all_turns = []
        self._is_async = self._check_if_async_client()
        
        # 初始化共享模型
        self._init_shared_model()
    
    def _check_if_async_client(self):
        """Check if client is AsyncOpenAI"""
        return hasattr(self.client, '__class__') and 'Async' in self.client.__class__.__name__
    
    @classmethod
    def _init_shared_model(cls):
        """初始化类级别共享的embedding模型（只加载一次）"""
        if cls._shared_retrieval_model is None:
            import threading
            if cls._model_lock is None:
                cls._model_lock = threading.Lock()
            if cls._encode_lock is None:
                cls._encode_lock = threading.Lock()
            
            with cls._model_lock:
                if cls._shared_retrieval_model is None:
                    import torch
                    device = 'cpu'
                    print(f"Loading shared SentenceTransformer model on {device}...")
                    cls._shared_retrieval_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    print("SentenceTransformer model loaded and shared across all agents")
    
    @property
    def retrieval_model(self):
        """返回共享的retrieval模型"""
        return self._shared_retrieval_model
    
    def _init_model(self):
        """保持向后兼容，实际使用共享模型"""
        self._init_shared_model()
        
    def add_memory_org(self, content: str) -> Dict[str, Any]:
        """添加记忆内容"""
        try:
            # 确保模型已初始化
            self._init_model()
            
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
                self.corpus_embeddings = self.retrieval_model.encode(self.all_turns, convert_to_tensor=True)
            
            return {"success": True, "total_turns": len(self.all_turns)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_haystack(self, haystack_sessions: List[List[Dict]], haystack_dates: List[str]) -> Dict[str, Any]:
        """处理haystack格式的记忆数据"""
        # 确保模型已初始化
        self._init_model()
        
        new_turns = []
        for session, date in zip(haystack_sessions, haystack_dates):
            for turn in session:
                if 'role' in turn and 'content' in turn:
                    new_turns.append(f"[{date}] {turn['role']}: {turn['content']}")
        
        self.all_turns.extend(new_turns)
        self.corpus_embeddings = self.retrieval_model.encode(self.all_turns, convert_to_tensor=True)
        
        return {"success": True, "turns_added": len(new_turns), "total_turns": len(self.all_turns)}
    
    def QA(self, query: str) -> str:
        """回答问题"""
        try:
            # 确保模型已初始化
            self._init_model()
            
            if self.corpus_embeddings is None or len(self.all_turns) == 0:
                return "I don't have any memory to answer questions from."
            
            # 步骤1: 检索相关记忆
            query_embedding = self.retrieval_model.encode(query[query.find(']')+1:].strip(), convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
            retrieved_turns = [self.all_turns[hit['corpus_id']] for hit in hits]
            
            # 步骤2: 提取结构化事实
            facts = self._extract_facts(query, retrieved_turns)
            
            # 步骤3: 生成最终答案
            answer = self._generate_answer(query, facts, retrieved_turns)

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
            self._init_model()
            
            if self.corpus_embeddings is None or len(self.all_turns) == 0:
                return "I don't have any memory to answer questions from."
            
            # 使用锁保护encode操作的线程安全
            import asyncio
            def encode_query():
                with EmergenceAgent._encode_lock:
                    return self.retrieval_model.encode(query[query.find(']')+1:].strip(), convert_to_tensor=True)
            
            query_embedding = await asyncio.to_thread(encode_query)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
            retrieved_turns = [self.all_turns[hit['corpus_id']] for hit in hits]
            
            facts = await self._extract_facts_async(query, retrieved_turns)
            answer = await self._generate_answer_async(query, facts, retrieved_turns)

            return answer.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def QA_batch_async(self, queries: List[str], batch_size: int = 32) -> List[str]:
        """异步批量回答问题"""
        results = []
        for query in queries:
            result = await self.QA_async(query)
            results.append(result)
        return results
    
    def _extract_facts(self, question: str, retrieved_turns: List[str]) -> str:
        """从检索到的对话中提取结构化事实"""
        summary_prompt = f"""
        You are a memory summarization assistant. Extract relevant facts to answer the question. Follow this chain-of-thought:
        1. Identify key events, dates, quantities, or named entities.
        2. Extract only information relevant to the question.
        3. Write the facts in structured bullet points.
        
        Question: {question}
        
        Messages:
        {json.dumps(retrieved_turns, indent=2)}
        
        Now extract the structured facts:
        -
        """
        summary_prompt = textwrap.dedent(summary_prompt)
        
        try:
            facts = self.client.chat.completions.create(
                model=MODEL_NAME_MAP.get(self.model_name, self.model_name),
                messages=[{"role": "system", "content": summary_prompt}],
                temperature=0.0,
                max_tokens=512
            ).choices[0].message.content
            return facts
        except Exception as e:
            print(f"Error extracting facts: {str(e)}")
            return f"Error extracting facts: {str(e)}"
    
    def _generate_answer(self, question: str, facts: str, retrieved_turns: List[str]) -> str:
        """使用提取的事实和检索到的对话生成最终答案"""
        answer_prompt = f"""
        You are a helpful assistant. Using both the extracted facts and the original conversation turns below,
        answer the question as accurately and concisely as possible.
        
        Extracted Facts:
        {facts}
        
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
            # 确保模型已初始化
            self._init_model()
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
                self.corpus_embeddings = self.retrieval_model.encode(self.all_turns, convert_to_tensor=True)
            
            return {"success": True, "total_turns": len(self.all_turns), "turns_added": turns_added}
            
        except Exception as e:
            return {"success": False, "error": str(e)}    
    async def _extract_facts_async(self, question: str, retrieved_turns: List[str]) -> str:
        """异步提取结构化事实"""
        summary_prompt = f"""
        You are a memory summarization assistant. Extract relevant facts to answer the question. Follow this chain-of-thought:
        1. Identify key events, dates, quantities, or named entities.
        2. Extract only information relevant to the question.
        3. Write the facts in structured bullet points.
        
        Question: {question}
        
        Messages:
        {json.dumps(retrieved_turns, indent=2)}
        
        Now extract the structured facts:
        -
        """
        summary_prompt = textwrap.dedent(summary_prompt)
        
        try:
            response = await self.client.chat.completions.create(
                model=MODEL_NAME_MAP.get(self.model_name, self.model_name),
                messages=[{"role": "system", "content": summary_prompt}],
                temperature=0.0,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error extracting facts: {str(e)}"
    
    async def _generate_answer_async(self, question: str, facts: str, retrieved_turns: List[str]) -> str:
        """异步生成最终答案"""
        answer_prompt = f"""
        You are a helpful assistant. Using both the extracted facts and the original conversation turns below,
        answer the question as accurately and concisely as possible.
        
        Extracted Facts:
        {facts}
        
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
            self._init_model()
            
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
                import asyncio
                import torch
                
                # 只编码新增的turns（增量编码）
                new_turns = self.all_turns[old_turns_count:]
                
                # 使用锁保护encode操作的线程安全
                def encode_with_lock():
                    with EmergenceAgent._encode_lock:
                        return self.retrieval_model.encode(new_turns, convert_to_tensor=True)
                
                new_embeddings = await asyncio.to_thread(encode_with_lock)
                
                # 拼接到现有embeddings
                if self.corpus_embeddings is None:
                    self.corpus_embeddings = new_embeddings
                else:
                    self.corpus_embeddings = torch.cat([self.corpus_embeddings, new_embeddings], dim=0)
            
            return {"success": True, "total_turns": len(self.all_turns), "turns_added": turns_added}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
