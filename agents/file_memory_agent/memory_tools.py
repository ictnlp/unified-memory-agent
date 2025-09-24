"""
Memory Tools for FileMemoryAgent
基于 MemU 的工具实现，使用 smolagents
"""

import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path

# 尝试导入 smolagents
try:
    from smolagents import tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    # 创建一个简单的装饰器作为 fallback
    def tool(func):
        func.is_tool = True
        return func

# 尝试导入 RecallAgent (如果可用)
try:
    from memu.memory import RecallAgent
    RECALL_AGENT_AVAILABLE = True
except ImportError:
    RECALL_AGENT_AVAILABLE = False
    RecallAgent = None


@tool
def search_file_content(file_paths: str, pattern: str, case_sensitive: bool = False) -> str:
    """
    Searches for regular expression patterns in file content.
    Must use this tool before final_answer.
    
    Args:
        file_paths (str): File path, JSON array of paths, or directory path
        pattern (str): Regular expression pattern to search for
        case_sensitive (bool): Whether search should be case sensitive
    """
    try:
        # 编译正则表达式
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        # 解析文件路径
        if file_paths.startswith('[') and file_paths.endswith(']'):
            # JSON数组格式
            paths = json.loads(file_paths)
        elif os.path.isdir(file_paths):
            # 目录路径，递归搜索
            paths = []
            for root, dirs, files in os.walk(file_paths):
                # 排除特定目录
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', '.env', 'embeddings'}]
                
                for file in files:
                    # 只搜索文本文件
                    if any(file.endswith(ext) for ext in ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.log', '.cfg', '.conf', '.ini']):
                        paths.append(os.path.join(root, file))
        else:
            # 单个文件路径
            paths = [file_paths]
        
        # 搜索结果
        total_matches = 0
        files_with_matches = 0
        results = []
        
        for path in paths:
            try:
                if not os.path.exists(path):
                    continue
                    
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                matches = []
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append((line_num, line.strip()))
                        total_matches += 1
                
                if matches:
                    files_with_matches += 1
                    results.append((path, matches))
                    
            except Exception as e:
                continue
        
        # 格式化输出
        output = f"🔍 Search Results for pattern: '{pattern}'\n"
        output += f"📊 Found {total_matches} total matches in {files_with_matches}/{len(paths)} files\n"
        output += f"🗂️  Total files searched: {len(paths)}\n"
        output += f"Case sensitive: {case_sensitive}\n"
        output += "-" * 60 + "\n\n"
        
        if not results:
            return output + "No matches found."
        
        for file_path, matches in results:
            output += f"📁 {file_path} ({len(matches)} matches)\n"
            for line_num, line in matches[:10]:  # 限制每个文件最多显示10个匹配
                output += f"  Line {line_num}: {line}\n"
            if len(matches) > 10:
                output += f"  ... and {len(matches) - 10} more matches\n"
            output += "\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error in file search: {str(e)}"


@tool
def retrieve_relevant_memories(memory_dir: str, character_name: str, query: str, top_k: int = 5) -> str:
    """
    Retrieves relevant memories using semantic search across all categories.
    Must use this tool before final_answer.
    
    Args:
        memory_dir (str): Path to the memory directory containing memory files
        character_name (str): Name of the character (e.g., "user")  
        query (str): The query to search for relevant memories
        top_k (int): Number of top relevant memories to return (default: 5)
    """
    try:
        # 如果有 RecallAgent，使用语义搜索
        if RECALL_AGENT_AVAILABLE:
            recall_agent = RecallAgent(memory_dir=memory_dir)
            result = recall_agent.retrieve_relevant_memories(character_name, query, top_k)
            
            if not result.get("success"):
                return f"❌ Retrieval failed: {result.get('error', 'Unknown error')}"
            
            results = result.get('results', [])
            total_candidates = result.get('total_candidates', 0)
            
            # 格式化输出
            output = f"🔍 Memory Retrieval Results for: '{query}'\n"
            output += f"📊 Found {total_candidates} candidates, showing top {len(results)}:\n"
            output += "-" * 60 + "\n\n"
            
            if not results:
                return output + "No relevant memories found."
            
            for i, item in enumerate(results, 1):
                category = item.get('category', 'unknown')
                score = item.get('semantic_score', 0.0)
                length = item.get('length', 0)
                line_number = item.get('line_number', 0)
                content = item.get('content', '')
                item_id = item.get('item_id', '')
                memory_id = item.get('memory_id', '')
                
                output += f"{i}. {category.upper()} (similarity: {score:.3f})\n"
                output += f"   📄 Length: {length} chars, Line: {line_number}\n"
                output += f"   🆔 Item ID: {item_id}\n"
                output += f"   🧠 Memory ID: {memory_id}\n"
                output += f"   💭 MEMORY CONTENT:\n"
                output += "   " + "="*50 + "\n"
                output += f"   {content}\n"
                output += "   " + "="*50 + "\n\n"
            
            return output
            
        else:
            # Fallback: 简单的文本搜索
            return _simple_memory_retrieval(memory_dir, character_name, query, top_k)
            
    except Exception as e:
        return f"❌ Error in memory retrieval: {str(e)}"


def _simple_memory_retrieval(memory_dir: str, character_name: str, query: str, top_k: int = 5) -> str:
    """简单的基于关键词的记忆检索作为 fallback"""
    try:
        memory_path = Path(memory_dir)
        if not memory_path.exists():
            return f"❌ Memory directory not found: {memory_dir}"
        
        # 查找所有记忆文件
        memory_files = list(memory_path.glob(f"{character_name}_*.md"))
        if not memory_files:
            return f"❌ No memory files found for {character_name}"
        
        results = []
        query_words = query.lower().split()
        
        for file_path in memory_files:
            category = file_path.stem.replace(f"{character_name}_", "")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析记忆项
                memory_items = re.findall(r'\[([^\]]+)\]\[mentioned at ([^\]]+)\] ([^\[\]]*?)(?=\s*\[|$)', content, re.DOTALL)
                
                for memory_id, date, memory_content in memory_items:
                    memory_content = memory_content.strip()
                    if not memory_content:
                        continue
                    
                    # 简单的关键词匹配评分
                    score = sum(1 for word in query_words if word in memory_content.lower())
                    if score > 0:
                        results.append({
                            'category': category,
                            'score': score / len(query_words),  # 归一化分数
                            'content': memory_content,
                            'memory_id': memory_id,
                            'date': date,
                            'length': len(memory_content)
                        })
                        
            except Exception as e:
                continue
        
        # 按分数排序并取前 top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]
        
        # 格式化输出
        output = f"🔍 Memory Retrieval Results for: '{query}'\n"
        output += f"📊 Found {len(results)} relevant memories (keyword-based search):\n"
        output += "-" * 60 + "\n\n"
        
        if not results:
            return output + "No relevant memories found."
        
        for i, item in enumerate(results, 1):
            output += f"{i}. {item['category'].upper()} (score: {item['score']:.3f})\n"
            output += f"   📄 Length: {item['length']} chars\n"
            output += f"   🧠 Memory ID: {item['memory_id']}\n"
            output += f"   📅 Date: {item['date']}\n"
            output += f"   💭 MEMORY CONTENT:\n"
            output += "   " + "="*50 + "\n"
            output += f"   {item['content']}\n"
            output += "   " + "="*50 + "\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error in simple memory retrieval: {str(e)}"


# 工具列表
memory_tools = [
    retrieve_relevant_memories,
    search_file_content,
]

__all__ = ['memory_tools', 'retrieve_relevant_memories', 'search_file_content']