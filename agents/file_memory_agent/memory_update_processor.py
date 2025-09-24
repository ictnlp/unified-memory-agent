"""
Step 4: Memory Update Processor
根据MemU的update_memory_with_suggestions实现
"""

import uuid
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple


class MemoryUpdateProcessor:
    """处理记忆更新的第四步：基于建议更新记忆类别"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.basic_memory_types = ["profile", "event", "episodic", "semantic"]
    
    def process(self, character_name: str, category: str, suggestion: str, 
                existing_content: str = "", session_date: str = None) -> Dict[str, Any]:
        """
        执行步骤4：更新记忆类别
        
        Args:
            character_name: 角色名称
            category: 记忆类别
            suggestion: 类别建议
            existing_content: 现有内容
            session_date: 会话日期
            
        Returns:
            包含操作结果的字典
        """
        try:
            if category not in self.basic_memory_types:
                return {
                    "success": False,
                    "error": f"Invalid category '{category}'. Available: {self.basic_memory_types}",
                }
            
            if not session_date:
                session_date = datetime.now().strftime("%Y-%m-%d")
            
            # 加载现有内容
            existing_memory_items = self._extract_memory_items_from_content(existing_content)
            formatted_existing_content = self._format_existing_content(existing_memory_items)
            
            # 分析记忆操作
            operation_response = self._analyze_memory_operation_from_suggestion(
                category, character_name, formatted_existing_content, suggestion
            )
            
            if not operation_response.strip():
                return {
                    "success": False,
                    "error": f"LLM returned empty operation analysis for {category}",
                }
            
            # 解析操作响应
            operation_list = self._parse_operation_response(operation_response)
            operation_executed, new_items = self._execute_operations(
                character_name, category, operation_list, session_date, existing_memory_items
            )
            
            return {
                "success": True,
                "character_name": character_name,
                "category": category,
                "operation_executed": operation_executed,
                "new_memory_items": new_items,
                "updated_content": self._format_memory_items(existing_memory_items),
                "message": f"Successfully performed {len(operation_executed)} memory operations for {category}",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _format_existing_content(self, existing_memory_items: List[Dict[str, str]]) -> str:
        """格式化现有内容为记忆项列表"""
        return "\n".join([
            f"[Memory ID: {item['memory_id']}] {item['content']}"
            for item in existing_memory_items
        ])
    
    def _analyze_memory_operation_from_suggestion(self, category: str, character_name: str, 
                                                  existing_content: str, suggestion: str) -> str:
        """分析记忆更新场景并确定应执行的操作，完全按照MemU的原始prompt"""
        
        # 完全按照MemU的原始prompt
        operation_prompt = f"""You are an expert in analyzing the following memory update scenario and determining the memory operations that should be performed.

Character: {character_name}
Memory Category: {category}

Existing Memory Items in {category}:
{existing_content if existing_content else "No existing content"}

Memory Update Suggestion:
{suggestion}

**CRITICAL REQUIREMENT: The object of memory operations must be SELF-CONTAINED MEMORY ITEMS**

**SELF-CONTAINED MEMORY REQUIREMENTS:**
- EVERY activity item must be complete and standalone
- ALWAYS include the full subject (do not use "she/he/they/it")
- NEVER use pronouns that depend on context (no "she", "he", "they", "it")
- Include specific names, places, dates, and full context in each item
- Each activity should be understandable without reading other items
- Include all relevant details, emotions, and outcomes in the activity description

**OPERATION TYPES:**
1. **ADD**: Add completely new memory items that doesn't exist in Existing Memory Items
2. **UPDATE**: Modify or enhance existing memory items with new details
3. **DELETE**: Remove outdated, incorrect, or irrelevant memory items
4. **TOUCH**: Touch memory items that already exists in current content (only for updating last-mentioned timestamp)

**ANALYSIS GUIDELINES:**
- Read the Memory Update Suggestion carefully to determine what new memory items are offered
- Read the Existing Memory Items to view all memory items that are already present
- Determine the most appropriate operation type FOR EACH NEW MEMORY ITEM based on the new information and existing content
- **Use ADD for:** New memory items that are not covered in existing content
- **Use UPDATE for:** New memory items that provide updated details for existing memory items
- **Use DELETE for:** Existing memory items that are outdated/incorrect based on new memory items
- **Use TOUCH for:** Existing memory items that already covers the new memory items adequately

**OUTPUT INSTRUCTIONS:**
- **IMPORTANT** Output ALL necessary memory operations. It is common that you should perform different operations for different specific memory items
- For ADD and UPDATE operations, provide the content of the new memory items following the self-contained memory requirements
- For UPDATE, DELETE, and TOUCH operations, provide the target memory IDs associated with the memory items
- If there are multiple actions for an operation type (e.g, multiple ADDs), output them separately, do not put them in a single **OPERATION:** block
- **IMPORTANT** If a memory item in suggestion uses modal adverbs (perhaps, probably, likely, etc.) to indicate an uncertain inference, keep the modal adverbs as-is in your output

**OUTPUT FORMAT:**

**OPERATION:** [ADD/UPDATE/DELETE/TOUCH]
- Target Memory ID: [Only if operation is UPDATE, DELETE, or TOUCH][Memory ID of the memory item that is the target of the operation]
- Memory Item Content: [Only if operation is ADD or UPDATE][Content of the new memory item]

**OPERATION:** [ADD/UPDATE/DELETE/TOUCH]
- Target Memory ID: [Only if operation is UPDATE, DELETE, or TOUCH][Memory ID of the memory item that is the target of the operation]
- Memory Item Content: [Only if operation is ADD or UPDATE][Content of the new memory item]

... other operations ...
"""
        
        # 调用LLM确定操作类型和内容
        operation_response = self.llm_client.chat.completions.create(
            model="azure-gpt-4_1",
            messages=[{"role": "user", "content": operation_prompt}],
            max_tokens=2048,
            temperature=0.1
        ).choices[0].message.content
        
        return operation_response
    
    def _parse_operation_response(self, response: str) -> List[Dict[str, Any]]:
        """解析LLM响应以提取操作信息"""
        lines = response.strip().split("\n")
        
        operation_list = []
        current_operation = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("**OPERATION:**"):
                operation = line.replace("**OPERATION:**", "").strip()
                if operation in ["ADD", "UPDATE", "DELETE", "TOUCH"]:
                    if current_operation:
                        operation_list.append(current_operation)
                    current_operation = {
                        "operation": operation,
                        "target_id": None,
                        "memory_content": None,
                    }
            
            if line.startswith("- Target Memory ID:") and current_operation:
                target_id = line.replace("- Target Memory ID:", "").strip()
                current_operation["target_id"] = target_id
            
            if line.startswith("- Memory Item Content:") and current_operation:
                memory_content = line.replace("- Memory Item Content:", "").strip()
                current_operation["memory_content"] = memory_content
        
        if current_operation:
            operation_list.append(current_operation)
        
        return operation_list
    
    def _execute_operations(self, character_name: str, category: str, operation_list: List[Dict[str, Any]], 
                           session_date: str, existing_items: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict]]:
        """执行列表中的所有操作"""
        
        all_items = existing_items.copy()
        new_items = []
        updated_items = []
        operation_executed = []
        
        for operation in operation_list:
            if operation["operation"] == "ADD":
                if not operation["memory_content"]:
                    continue
                
                memory_id = self._generate_memory_id()
                memory_item = {
                    "memory_id": memory_id,
                    "mentioned_at": session_date,
                    "content": operation["memory_content"],
                    "links": "",
                }
                
                all_items.append(memory_item)
                new_items.append(memory_item)
                updated_items.append(memory_item)
                operation_executed.append(operation)
            
            elif operation["operation"] == "UPDATE":
                if not operation["target_id"] or not operation["memory_content"]:
                    continue
                
                for item in all_items:
                    if item["memory_id"] == operation["target_id"]:
                        item["content"] = operation["memory_content"]
                        updated_items.append(item)
                        break
                
                operation_executed.append(operation)
            
            elif operation["operation"] == "DELETE":
                if not operation["target_id"]:
                    continue
                
                all_items = [item for item in all_items if item["memory_id"] != operation["target_id"]]
                operation_executed.append(operation)
            
            elif operation["operation"] == "TOUCH":
                if not operation["target_id"]:
                    continue
                
                # 对于TOUCH操作，应该更新"updated_at"
                for item in all_items:
                    if item["memory_id"] == operation["target_id"]:
                        # 这里可以添加更新时间戳的逻辑
                        pass
                
                operation_executed.append(operation)
        
        # 更新existing_items引用
        existing_items[:] = all_items
        
        return operation_executed, new_items
    
    def _extract_memory_items_from_content(self, content: str) -> List[Dict[str, str]]:
        """从内容中提取记忆项，支持旧的和新的时间戳格式"""
        items = []
        if not content:
            return items
            
        lines = content.split("\n")
        
        for line in lines:
            line = line.strip()
            
            pattern = r"^\[([^\]]+)\]\[mentioned at ([^\]]+)\]\s*(.*?)(?:\s*\[([^\]]*)\])?$"
            match = re.match(pattern, line)
            if match:
                memory_id = match.group(1)
                mentioned_at = match.group(2)
                clean_content = match.group(3).strip()
                links = match.group(4) if match.group(4) else ""
                
                if memory_id and clean_content:
                    items.append({
                        "memory_id": memory_id,
                        "mentioned_at": mentioned_at,
                        "content": clean_content,
                        "links": links,
                    })
        
        return items
    
    def _format_memory_items(self, items: List[Dict[str, str]]) -> str:
        """将记忆项格式化为字符串"""
        return "\n".join([
            f"[{item['memory_id']}][mentioned at {item['mentioned_at']}] {item['content']} [{item['links']}]"
            for item in items
        ])
    
    def _generate_memory_id(self) -> str:
        """生成唯一的记忆ID"""
        return str(uuid.uuid4()).replace('-', '')[:8]