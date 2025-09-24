"""
EmergenceAgent - 基于RAG的简化记忆代理
基于emergence_simple_fast/main.py的核心实现
"""

import json
import textwrap
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

from .base_agent import BaseAgent


class EmergenceAgent(BaseAgent):
    """基于语义检索和事实提取的简化记忆代理"""
    
    def __init__(self, client: OpenAI, top_k: int = 42):
        super().__init__(client)
        
        # 延迟初始化嵌入模型，避免多进程CUDA冲突
        self.retrieval_model = None
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.top_k = top_k
        
        # 记忆存储
        self.corpus_embeddings = None
        self.all_turns = []
    
    def _init_model(self):
        """延迟初始化嵌入模型，使用CPU避免多进程CUDA冲突"""
        if self.retrieval_model is None:
            import torch
            # 强制使用CPU，避免多进程CUDA冲突（单进程模式）
            device = 'cpu'
            self.retrieval_model = SentenceTransformer(self.embedding_model_name, device=device)
        
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
    
    def QA_batch(self, queries: List[str]) -> List[str]:
        """批量回答问题"""
        return [self.QA(query) for query in queries]
    
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
                model="azure-gpt-4_1",
                messages=[{"role": "system", "content": summary_prompt}],
                temperature=0.0,
                max_tokens=512
            ).choices[0].message.content
            return facts
        except Exception as e:
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
                model="azure-gpt-4_1", 
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=256
            ).choices[0].message.content
            return answer
        except Exception as e:
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