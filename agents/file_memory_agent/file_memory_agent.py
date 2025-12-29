import os
import json
import uuid
import tempfile
import shutil
import re
from typing import List, Dict, Any
from pathlib import Path

from ..base_agent import BaseAgent, MODEL_NAME_MAP
from .combined_processor import CombinedProcessor
from .suggestions_processor import SuggestionsProcessor
from .memory_update_processor import MemoryUpdateProcessor

class MemoryProcessor:
    """完整的记忆处理器，实现MemU的3步记忆生成流程（优化版）"""
    
    def __init__(self, llm_client, memory_dir: str):
        self.llm_client = llm_client
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.combined_processor = CombinedProcessor(llm_client)
        self.suggestions_processor = SuggestionsProcessor(llm_client)
        self.memory_update_processor = MemoryUpdateProcessor(llm_client)
    
    def process_conversation(self, conversation_text: str, character_name: str = "user") -> Dict[str, Any]:
        """执行完整的MemU 3步记忆生成流程（优化版）"""
        try:
            files_generated = []
            session_date = "2023-05-20"  # 可以从conversation_text中提取或使用当前日期
            
            print(f"🚀 Starting MemU 3-step memory processing for {character_name}")
            
            # Step 1: Combined Activity Memory & Theory of Mind - 调用LLM同时生成活动记忆和心理理论分析
            print("🔧 Step 1: Processing activity memory and theory of mind analysis")
            step1_result = self.combined_processor.process(
                character_name=character_name,
                content=conversation_text,
                session_date=session_date
            )
            
            if not step1_result.get("success"):
                return {"success": False, "error": f"Step 1 failed: {step1_result.get('error')}", "files_generated": []}
            
            # 保存活动记忆到MemU格式的.md文件
            activity_file = self._save_activity_memory(step1_result, character_name)
            if activity_file:
                files_generated.append(activity_file)
            
            activity_items = step1_result.get("activity_items", [])
            theory_of_mind_items = step1_result.get("theory_of_mind_items", [])
            print(f"    Generated {len(activity_items)} activity memory items")
            print(f"    Generated {len(theory_of_mind_items)} theory of mind items")
            
            # Step 2: Generate Memory Suggestions - 调用LLM生成各类别建议
            print("🔧 Step 2: Generating memory suggestions")
            combined_memory_items = activity_items + theory_of_mind_items
            
            step2_result = self.suggestions_processor.process(
                character_name=character_name,
                new_memory_items=combined_memory_items
            )
            
            if not step2_result.get("success"):
                return {"success": False, "error": f"Step 2 failed: {step2_result.get('error')}", "files_generated": files_generated}
            
            suggestions = step2_result.get("suggestions", {})
            print(f"    Generated suggestions for {len(suggestions)} categories")
            
            # Step 3: Update Memory Categories - 调用LLM执行记忆操作
            print("🔧 Step 3: Updating memory categories")
            for category, suggestion in suggestions.items():
                if suggestion and suggestion.strip():
                    print(f"    Updating category: {category}")
                    
                    # 读取现有内容
                    existing_content = self._read_category_content(character_name, category)
                    
                    # 调用记忆更新处理器
                    step3_result = self.memory_update_processor.process(
                        character_name=character_name,
                        category=category,
                        suggestion=suggestion,
                        existing_content=existing_content,
                        session_date=session_date
                    )
                    
                    if step3_result.get("success"):
                        # 保存更新的内容到MemU格式
                        updated_file = self._save_category_content(
                            character_name, category, step3_result.get("updated_content", "")
                        )
                        if updated_file:
                            files_generated.append(updated_file)
                        
                        new_items_count = len(step3_result.get("new_memory_items", []))
                        print(f"        Added {new_items_count} new memory items to {category}")
            
            print(f"✅ MemU 3-step processing completed: {len(files_generated)} files generated")
            
            return {
                "success": True,
                "iterations": 1,
                "files_generated": files_generated,
                "step1_activity_items": len(activity_items),
                "step1_theory_items": len(theory_of_mind_items),
                "step2_suggestions": len(suggestions),
                "step3_categories": len([s for s in suggestions.values() if s.strip()])
            }
            
        except Exception as e:
            print(f"❌ Error in MemU processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_generated": files_generated
            }
    
    def _save_activity_memory(self, step1_result: Dict[str, Any], character_name: str) -> str:
        """保存步骤1的活动记忆结果到MemU格式的.md文件"""
        try:
            activity_file = self.memory_dir / f"{character_name}_activity.md"
            
            # 如果文件已存在，追加内容；否则创建新文件
            mode = 'a' if activity_file.exists() else 'w'
            
            with open(activity_file, mode, encoding='utf-8') as f:
                if mode == 'w':
                    # 新文件，不需要额外的头部
                    pass
                
                # 从合并结果中获取活动记忆的格式化内容
                formatted_content = step1_result.get("activity_formatted_content", "") or step1_result.get("formatted_content", "")
                if formatted_content:
                    f.write(formatted_content + "\n")
                    
            return str(activity_file)
        except Exception as e:
            print(f"Error saving activity memory: {e}")
            return None
    
    def _read_category_content(self, character_name: str, category: str) -> str:
        """读取指定类别的现有内容（MemU格式）"""
        try:
            category_file = self.memory_dir / f"{character_name}_{category}.md"
            if not category_file.exists():
                return ""
            
            with open(category_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return content
            
        except Exception as e:
            print(f"Error reading category content for {category}: {e}")
            return ""
    
    def _save_category_content(self, character_name: str, category: str, content: str) -> str:
        """保存类别内容到MemU格式的.md文件"""
        try:
            category_file = self.memory_dir / f"{character_name}_{category}.md"
            
            with open(category_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return str(category_file)
        except Exception as e:
            print(f"Error saving {category} content: {e}")
            return None
    

class MemoryRetriever:
    """记忆检索工具，模拟smolagent的工具，支持MemU格式的.md文件"""
    
    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
    
    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> str:
        """检索相关记忆 - 模拟smolagent的retrieve_relevant_memories工具"""
        if not self.memory_dir.exists():
            return "No memory directory found."
        
        all_memories = []
        
        # 遍历MemU格式的.md文件
        for md_file in self.memory_dir.glob("user_*.md"):
            category = md_file.stem.replace("user_", "")  # 从user_activity.md提取activity
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 按行分割，每行是一个记忆项
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if line:  # 非空行
                        # 简单的相关性评分（基于关键词匹配）
                        query_words = query.lower().split()
                        line_lower = line.lower()
                        score = sum(line_lower.count(word) for word in query_words)
                        
                        all_memories.append({
                            "file": str(md_file),
                            "category": category,
                            "line_number": line_num,
                            "content": line,
                            "score": score
                        })
            except Exception as e:
                continue
        
        # 按分数排序并返回top_k
        all_memories.sort(key=lambda x: x["score"], reverse=True)
        top_memories = all_memories[:top_k]
        
        result = f"=== Top {len(top_memories)} Relevant Memories ===\n\n"
        for i, mem in enumerate(top_memories, 1):
            result += f"**Memory {i} (Category: {mem['category']}, Score: {mem['score']})**\n"
            result += f"File: {mem['file']} (Line {mem['line_number']})\n"
            result += f"Content: {mem['content'][:500]}...\n\n"
        
        return result if top_memories else "No relevant memories found."
    
    def search_file_content(self, pattern: str, case_sensitive: bool = False) -> str:
        """搜索文件内容 - 模拟smolagent的search_file_content工具"""
        if not self.memory_dir.exists():
            return "No memory directory found."
        
        flags = 0 if case_sensitive else re.IGNORECASE
        results = []
        
        # 遍历MemU格式的.md文件
        for md_file in self.memory_dir.glob("user_*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, flags):
                            results.append({
                                "file": str(md_file),
                                "line_number": line_num,
                                "line_content": line.strip()
                            })
            except Exception as e:
                continue
        
        if not results:
            return f"No matches found for pattern: {pattern}"
        
        result_text = f"=== Search Results for '{pattern}' ===\n\n"
        for result in results[:20]:  # 限制结果数量
            result_text += f"File: {result['file']}\n"
            result_text += f"Line {result['line_number']}: {result['line_content']}\n\n"
        
        if len(results) > 20:
            result_text += f"... and {len(results) - 20} more matches.\n"
        
        return result_text

class FileMemoryAgent(BaseAgent):
    """基于文件存储的记忆代理，完整实现MemU的记忆生成和检索流程"""
    
    def __init__(self, client=None, model_name: str = "gpt4.1", task_type: str = None, task_id: str = None):
        super().__init__(client, model_name)
        
        # 根据task信息创建记忆目录（遵循MemU格式）
        if task_type and task_id:
            # 检查task_id是否已经包含task_type
            if task_id.startswith(task_type):
                # task_id已经包含task_type，直接使用
                memory_dir_name = f"memory_{task_id}"
            else:
                # task_id不包含task_type，组合两者
                memory_dir_name = f"memory_{task_type}_{task_id}"
        else:
            # 如果没有提供task信息，使用session ID
            session_id = uuid.uuid4().hex[:8]
            memory_dir_name = f"memory_session_{session_id}"
        
        # 在tmp目录下创建记忆目录
        # base_tmp_dir = tempfile.gettempdir()
        base_tmp_dir = "tmp"
        self.memory_dir = Path(base_tmp_dir) / memory_dir_name
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 不再在MemoryProcessor中创建session目录，直接使用记忆目录
        self.memory_processor = MemoryProcessor(self.client, str(self.memory_dir))
        
        print(f"FileMemoryAgent initialized with memory directory: {self.memory_dir}")
    
    def add_memory(self, chunk: str):
        """添加记忆块 - 使用完整的4步记忆生成流程"""
        try:
            result = self.memory_processor.process_conversation(chunk, "user")
            
            if result.get("success"):
                files_count = len(result.get('files_generated', []))
                print(f"Memory processed successfully: {files_count} files generated")
            else:
                print(f"Failed to process memory: {result.get('error')}")
                
        except Exception as e:
            print(f"Error adding memory: {e}")
    
    def QA(self, query: str) -> str:
        """使用 smolagents 进行问答 - 严格按照 MemU 的 run_agent_qa.py 方式"""
        try:
            # 尝试导入 smolagents
            try:
                from smolagents import ToolCallingAgent, OpenAIServerModel
                from .memory_tools import memory_tools
                SMOLAGENTS_AVAILABLE = True
            except ImportError:
                print("Warning: smolagents not available, falling back to simple QA")
                return self._fallback_qa(query)
            
            # 创建 OpenAI 模型 (按照 MemU 的配置)
            model = OpenAIServerModel(
                model_id="azure-gpt-4_1",  # 使用与 MemU 相同的模型
                temperature=0.0,
                api_base="http://api-hub.inner.chj.cloud/llm-gateway/v1",
                api_key="sk-",
                client_kwargs={
                    "default_headers": {
                        "BCS-APIHub-RequestId": str(uuid.uuid4()),
                        "X-CHJ-GWToken": os.getenv("X_CHJ_GWTOKEN"),  # 环境变量名改为X_CHJ_GWTOKEN
                        "X-CHJ-GW-SOURCE": os.getenv("X_CHJ_GW_SOURCE"),  # 也改为X_CHJ_GW_SOURCE
                    }
                }
            )
            
            # 创建 ToolCallingAgent (按照 MemU 的配置)
            agent = ToolCallingAgent(
                tools=memory_tools, 
                model=model, 
                stream_outputs=False, 
                final_answer_checks=[], 
                return_full_result=True, 
                max_steps=100
            )
            
            # 按照 MemU 的指令格式
            folder_path = str(self.memory_dir)
            instruct = f'''Please search for relevant information in the folder {folder_path} and answer the following question:
{query}
Noted that at least you need to use retrieve_relevant_memories and search_file_content.'''
            
            # 运行 agent
            outputs = agent.run(instruct)
            
            # 返回最终答案
            return outputs.output if hasattr(outputs, 'output') else str(outputs)
            
        except Exception as e:
            print(f"Error in smolagents QA: {e}")
            return self._fallback_qa(query)
    
    def _fallback_qa(self, query: str) -> str:
        """fallback QA 方法，当 smolagents 不可用时使用"""
        try:
            from .memory_tools import retrieve_relevant_memories, search_file_content
            
            folder_path = str(self.memory_dir)
            
            # Step 1: Retrieve relevant memories
            relevant_memories = retrieve_relevant_memories(folder_path, "user", query, 5)
            
            # Step 2: Search file content for specific patterns
            search_results = search_file_content(folder_path, query)
            
            # Step 3: 使用检索到的信息回答问题
            qa_prompt = f"""Please search for relevant information and answer the following question:

Question: {query}

=== Retrieved Relevant Memories ===
{relevant_memories}

=== File Search Results ===
{search_results}

Based on the above information, please provide a comprehensive and accurate answer to the question. If the information is insufficient, please indicate what additional information would be needed."""

            model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": qa_prompt}],
                max_tokens=1024,
                temperature=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return self._handle_api_error(e, query)
    
    def QA_batch(self, query_list: List[str]) -> List[str]:
        """批量回答问题 - 每个问题独立使用 smolagents 处理"""
        try:
            responses = []
            
            for query in query_list:
                try:
                    # 每个问题独立调用 QA 方法
                    response = self.QA(query)
                    responses.append(response)
                    
                except Exception as e:
                    error_msg = self._handle_api_error(e, query)
                    responses.append(error_msg)
            
            return responses
            
        except Exception as e:
            error_msg = self._handle_api_error(e, f"Batch queries: {query_list}")
            return [error_msg] * len(query_list)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if not self.memory_dir.exists():
            return {"memory_files": 0, "total_size": 0, "categories": {}}
        
        stats = {"categories": {}, "total_files": 0, "total_size": 0}
        
        # 检查MemU格式的.md文件
        for md_file in self.memory_dir.glob("user_*.md"):
            category = md_file.stem.replace("user_", "")  # 从user_activity.md提取activity
            file_size = md_file.stat().st_size
            
            stats["categories"][category] = {
                "files": 1,
                "size": file_size
            }
            stats["total_files"] += 1
            stats["total_size"] += file_size
        
        stats.update({
            "memory_dir": str(self.memory_dir),
        })
        
        return stats
    
    def cleanup(self):
        """清理临时文件"""
        try:
            if self.memory_dir.exists():
                shutil.rmtree(self.memory_dir)
                print(f"Cleaned up memory directory: {self.memory_dir}")
        except Exception as e:
            print(f"Error cleaning up: {e}")
    
    # def __del__(self):
    #     """析构函数，自动清理"""
    #     try:
    #         self.cleanup()
    #     except:
    #         pass  # 忽略清理错误