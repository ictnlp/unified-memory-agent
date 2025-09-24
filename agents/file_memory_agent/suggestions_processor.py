"""
Step 3: Memory Suggestions Processor
根据MemU的generate_suggestions实现
"""

from typing import Dict, List, Any


class SuggestionsProcessor:
    """处理记忆建议生成的第三步：为不同类别生成建议"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        # 基本记忆类型（排除activity）
        self.basic_memory_types = ["profile", "event", "episodic", "semantic"]
    
    def process(self, character_name: str, new_memory_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        执行步骤3：生成记忆建议
        
        Args:
            character_name: 角色名称
            new_memory_items: 新的记忆项列表（来自步骤1和2）
            
        Returns:
            包含各类别建议的处理结果
        """
        try:
            if not new_memory_items:
                return {"success": False, "error": "No memory items provided"}
            
            available_categories = self._get_available_categories(character_name)
            if not available_categories:
                return {"success": False, "error": "No available categories found"}
            
            # 转换记忆项为文本用于分析
            memory_items_text = "\n".join([
                f"- {item['content']}"
                for item in new_memory_items
            ])
            
            # 使用LLM分析并生成建议
            suggestions_prompt = self._create_suggestions_prompt(
                character_name, memory_items_text, available_categories
            )
            
            response = self.llm_client.chat.completions.create(
                model="azure-gpt-4_1",
                messages=[{"role": "user", "content": suggestions_prompt}],
                max_tokens=2048,
                temperature=0.1
            ).choices[0].message.content
            
            if not response.strip():
                return {"success": False, "error": "LLM returned empty suggestions"}
            
            # 解析文本响应
            suggestions = self._parse_suggestions_from_text(
                response.strip(), available_categories, new_memory_items
            )
            
            return {
                "success": True,
                "character_name": character_name,
                "suggestions": suggestions,
                "categories_analyzed": available_categories,
                "message": f"Generated self-contained suggestions for {len(suggestions)} categories based on {len(new_memory_items)} memory items",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_available_categories(self, character_name: str) -> List[str]:
        """获取可用的类别（排除activity）"""
        return [category for category in self.basic_memory_types if category != "activity"]
    
    def _create_suggestions_prompt(self, character_name: str, memory_items_text: str, available_categories: List[str]) -> str:
        """创建建议生成的prompt，完全按照MemU的原始prompt"""
        
        # 完全按照MemU的原始prompt
        suggestions_prompt = f"""You are an expert in analyzing the provided memory items for {character_name} and suggesting the memory items that should be added to each memory category.

New Memory Items:
{memory_items_text}

Available Categories: {', '.join(available_categories)}

**CRITICAL REQUIREMENT: Suggestions must be SELF-CONTAINED MEMORY ITEMS**

**SELF-CONTAINED MEMORY REQUIREMENTS:**
- EVERY activity item must be complete and standalone
- ALWAYS include the full subject (do not use "she/he/they/it")
- NEVER use pronouns that depend on context (no "she", "he", "they", "it")
- Include specific names, places, dates, and full context in each item
- Each activity should be understandable without reading other items
- Include all relevant details, emotions, and outcomes in the activity description

**CATEGORY-SPECIFIC REQUIREMENTS:**

For each category, analyze the new memory items and suggest what specific information should be extracted and added to that category:

- **activity**: Detailed description of the conversation, including the time, place, and people involved
- **profile**: ONLY basic personal information (age, location, occupation, education, family status, demographics) - EXCLUDE events, activities, things they did
- **event**: Specific events, dates, milestones, appointments, meetings, activities with time references
- **Other categories**: Relevant information for each specific category

**CRITICAL DISTINCTION - Profile vs Activity/Event:**
- Profile (GOOD): "Alice lives in San Francisco", "Alice is 28 years old", "Alice works at TechFlow Solutions"
- Profile (BAD): "Alice went hiking" (this is activity), "Alice attended workshop" (this is event)
- Activity/Event (GOOD): "Alice went hiking in Blue Ridge Mountains", "Alice attended photography workshop"

**SUGGESTION REQUIREMENTS:**
- Specify that memory items should include "{character_name}" as the subject
- Mention specific names, places, titles, and dates that should be included
- Ensure suggestions lead to complete, self-contained memory items
- Avoid suggesting content that would result in pronouns or incomplete sentences
- For profile: Focus ONLY on stable, factual, demographic information
- If one input memory item involves information belongs to multiple categories, you should reasonable seperete the information and provide suggestions to all involved categories
- **IMPORTANT** If the input memory item use modal adverbs (perhaps, probably, likely, etc.) to indicate an uncertain inference, keep the modal adverbs as-is in your suggestions

**OUTPUT INSTRUCTIONS:**
- **IMPORTANT** NEVER suggest categories that are not in the Available Categories
- Only output categories where there are suggestions for new memory items

**OUTPUT FORMAT:**

**Category: [category_name]**
- Suggestion: [What specific self-contained content should be added to this category, ensuring full subjects and complete context]
- Suggestion: [What specific self-contained content should be added to this category, ensuring full subjects and complete context]

**Category: [category_name]**
- Suggestion: [What specific self-contained content should be added to this category, ensuring full subjects and complete context]

... other categories ...
"""
        
        return suggestions_prompt
    
    def _parse_suggestions_from_text(self, response_text: str, available_categories: List[str], new_memory_items: List[Dict[str, str]]) -> Dict[str, str]:
        """从文本格式响应中解析建议"""
        suggestions = {}
        
        try:
            lines = response_text.split("\n")
            current_category = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("**Category:") and line.endswith("**"):
                    category_name = line.replace("**Category:", "").replace("**", "").strip()
                    if category_name in available_categories:
                        current_category = category_name
                        suggestions[current_category] = ""
                    else:
                        current_category = None
                elif current_category and line.startswith("- Suggestion:"):
                    suggestion_text = line.replace("- Suggestion:", "").strip()
                    suggestions[current_category] += f"{suggestion_text}\n"
            
            # 清理空建议
            suggestions = {k: v.strip() for k, v in suggestions.items() if v.strip()}
            
        except Exception as e:
            print(f"Failed to parse suggestions from text: {repr(e)}")
            
        return suggestions