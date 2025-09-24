"""
Combined Activity and Theory of Mind Processor
将步骤1和步骤2合并为一个步骤：同时生成活动记忆和心理理论分析
"""

import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Any


class CombinedProcessor:
    """合并的处理器：同时处理活动记忆和心理理论分析"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def process(self, character_name: str, content: str, session_date: str = None) -> Dict[str, Any]:
        """
        执行合并的步骤：同时生成活动记忆和心理理论分析
        
        Args:
            character_name: 角色名称
            content: 原始对话内容
            session_date: 会话日期
            
        Returns:
            包含活动记忆项和心理理论项的处理结果
        """
        try:
            if not session_date:
                session_date = datetime.now().strftime("%Y-%m-%d")
            
            # 使用LLM同时生成活动记忆和心理理论分析
            llm_response = self._process_with_combined_llm(character_name, content, session_date)
            
            if not llm_response.strip():
                return {"success": False, "error": "LLM returned empty response"}
            
            # 解析LLM响应
            activity_content, theory_of_mind_content, reasoning_process = self._parse_combined_response(llm_response)
            
            # 处理活动记忆项
            activity_items = []
            activity_formatted_content = ""
            if activity_content.strip():
                activity_items, activity_formatted_content = self._add_memory_ids_with_timestamp(
                    activity_content, session_date
                )
            
            # 处理心理理论项
            theory_of_mind_items = []
            if theory_of_mind_content.strip():
                theory_of_mind_items = self._parse_theory_of_mind_items(
                    theory_of_mind_content, session_date
                )
            
            return {
                "success": True,
                "character_name": character_name,
                "session_date": session_date,
                # 活动记忆相关
                "activity_items": activity_items,
                "activity_formatted_content": activity_formatted_content,
                "activity_items_count": len(activity_items),
                # 心理理论相关
                "theory_of_mind_items": theory_of_mind_items,
                "theory_of_mind_items_count": len(theory_of_mind_items),
                "reasoning_process": reasoning_process,
                # 合并的记忆项
                "memory_items": activity_items + theory_of_mind_items,
                "formatted_content": activity_formatted_content,  # 保持兼容性
                "message": f"Successfully generated {len(activity_items)} activity items and {len(theory_of_mind_items)} theory of mind items for {character_name}",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_with_combined_llm(self, character_name: str, content: str, session_date: str) -> str:
        """使用LLM同时处理活动记忆和心理理论分析"""
        
        user_name = character_name
        
        # 合并的prompt，同时处理活动记忆和心理理论分析
        combined_prompt = f"""You are processing conversation content for {user_name} on {session_date}. You need to perform TWO tasks simultaneously:

**TASK 1: Generate Activity Memory Items**
**TASK 2: Generate Theory of Mind Analysis**

Raw content to process:
{content}

**TASK 1: ACTIVITY MEMORY FORMATTING**

**CRITICAL REQUIREMENT: GROUP RELATED CONTENT INTO MEANINGFUL ACTIVITIES**

Transform this raw content into properly formatted activity memory items following these rules:

**MEANINGFUL ACTIVITY GROUPING REQUIREMENTS:**
- Group related sentences/statements into single, comprehensive activity descriptions
- Each activity should be a complete, self-contained description of what happened
- Combine related dialogue, actions, and context into cohesive activity blocks
- Only create separate items for genuinely different activities or topics
- Each activity item should tell a complete "story" or "event"

**SELF-CONTAINED MEMORY REQUIREMENTS:**
- EVERY activity item must be complete and standalone
- ALWAYS include the full subject (do not use "she/he/they/it")
- NEVER use pronouns that depend on context (no "she", "he", "they", "it")
- Include specific names, places, dates, and full context in each item
- Each activity should be understandable without reading other items
- Include all relevant details, emotions, and outcomes in the activity description

**FORMAT REQUIREMENTS:**
1. Each line = one complete, meaningful activity (may include multiple related sentences)
2. NO markdown headers, bullets, numbers, or structure
3. Write in plain text only
4. Focus on comprehensive, meaningful activity descriptions
5. Use specific names, titles, places, and dates
6. Each line ends with a period

**TASK 2: THEORY OF MIND ANALYSIS**

Analyze the conversation to infer information that is not explicitly mentioned but the character might have meant to express or the listener can reasonably deduce.

**INFERENCE GUIDELINES:**
- Leverage your reasoning skills to infer information not explicitly mentioned
- Use modal adverbs (perhaps, probably, likely, etc.) to indicate confidence level
- DO NOT repeat information that will be included in the activity items
- Focus on implicit emotions, motivations, relationships, preferences, etc.

**SELF-CONTAINED INFERENCE REQUIREMENTS:**
- Plain text only, no markdown grammar
- EVERY inference item must be complete and standalone
- ALWAYS include the full subject (do not use "she/he/they/it")
- NEVER use pronouns that depend on context
- Include specific names, places, dates, and full context in each item
- Each inference should be understandable without reading other items
- Use words like "perhaps" or "maybe" to indicate uncertainty

**OUTPUT FORMAT:**

**ACTIVITY ITEMS:**
[One complete, meaningful activity per line, no markdown headers, no structure, no numbering, no bullet points, ends with a period]

**REASONING PROCESS:**
[Your reasoning process for what kind of implicit information can be hidden behind the conversation, what are the evidences, how you get to your conclusion, and how confident you are.]

**INFERENCE ITEMS:**
[One piece of inference per line, no markdown headers, no structure, no numbering, no bullet points, ends with a period]
[After carefully reasoning, if you determine that there is no implicit information that can be inferred from the conversation beyond the explicit information already mentioned in the activity items, you can leave this section empty.]

**EXAMPLES:**

**ACTIVITY ITEMS:**
{character_name} attended a LGBTQ support group where {character_name} heard inspiring transgender stories and felt happy, thankful, accepted, and gained courage to embrace {character_name}'s true self.
{character_name} discussed future career plans with Melanie, expressing keen interest in counseling and mental health work to support people with similar issues, and Melanie encouraged {character_name} saying {character_name} would be a great counselor due to {character_name}'s empathy and understanding.

**REASONING PROCESS:**
Based on the conversation, I can infer that the character shows signs of being comfortable with LGBTQ identity and has strong empathy toward others facing similar challenges. This suggests a personal connection to these issues.

**INFERENCE ITEMS:**
{character_name} may have personal experience with gender identity questions or LGBTQ issues based on {character_name}'s strong emotional connection to the support group stories.
{character_name} perhaps values helping others as a core personal motivation given {character_name}'s career interest in counseling.

Now process the content:

"""
        
        # 调用LLM处理合并任务
        response = self.llm_client.chat.completions.create(
            model="azure-gpt-4_1",
            messages=[{"role": "user", "content": combined_prompt}],
            max_tokens=3072,  # 增加token限制因为需要处理更多内容
            temperature=0.1
        ).choices[0].message.content
        
        return response
    
    def _parse_combined_response(self, response_text: str) -> Tuple[str, str, str]:
        """解析合并响应，分离活动记忆、心理理论和推理过程"""
        
        activity_content = ""
        theory_of_mind_content = ""
        reasoning_process = ""
        
        try:
            lines = response_text.split("\n")
            
            current_section = ""
            
            for line in lines:
                line_stripped = line.strip()
                
                # 检测section headers
                if (line_stripped.upper().startswith("**ACTIVITY ITEMS:") or 
                    line_stripped.startswith("**") and "ACTIVITY ITEMS" in line_stripped.upper()):
                    current_section = "activity"
                    continue
                elif (line_stripped.upper().startswith("**REASONING PROCESS:") or 
                      line_stripped.startswith("**") and "REASONING PROCESS" in line_stripped.upper()):
                    current_section = "reasoning"
                    continue
                elif (line_stripped.upper().startswith("**INFERENCE ITEMS:") or 
                      line_stripped.startswith("**") and "INFERENCE ITEMS" in line_stripped.upper()):
                    current_section = "theory_of_mind"
                    continue
                
                # 跳过其他section headers
                if line_stripped.startswith("**") and line_stripped.endswith("**"):
                    continue
                
                # 添加内容到相应section
                if current_section == "activity" and line_stripped:
                    if activity_content:
                        activity_content += "\n" + line_stripped
                    else:
                        activity_content = line_stripped
                        
                elif current_section == "reasoning" and line_stripped:
                    if reasoning_process:
                        reasoning_process += "\n" + line_stripped
                    else:
                        reasoning_process = line_stripped
                        
                elif current_section == "theory_of_mind" and line_stripped:
                    if theory_of_mind_content:
                        theory_of_mind_content += "\n" + line_stripped
                    else:
                        theory_of_mind_content = line_stripped
                        
        except Exception as e:
            print(f"Failed to parse combined response: {repr(e)}")
        
        return activity_content, theory_of_mind_content, reasoning_process
    
    def _add_memory_ids_with_timestamp(self, content: str, session_date: str) -> Tuple[List[Dict], str]:
        """
        为活动记忆项添加记忆ID和时间戳
        格式: [memory_id][mentioned at {session_date}] {content}
        """
        if not content.strip():
            return [], ""
        
        lines = content.split("\n")
        processed_items = []
        plain_memory_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # 只处理非空行
                # 为该行生成新的唯一记忆ID
                memory_id = self._generate_memory_id()
                # 格式: [memory_id][mentioned at {session_date}] {content} [links]
                processed_items.append({
                    "memory_id": memory_id,
                    "mentioned_at": session_date,
                    "content": line,
                    "links": "",
                })
                plain_memory_lines.append(
                    f"[{memory_id}][mentioned at {session_date}] {line} []"
                )
        
        plain_memory_text = "\n".join(plain_memory_lines)
        return processed_items, plain_memory_text
    
    def _parse_theory_of_mind_items(self, content: str, session_date: str) -> List[Dict]:
        """解析心理理论项"""
        
        theory_of_mind_items = []
        
        if not content.strip():
            return theory_of_mind_items
        
        lines = content.split("\n")
        
        for line in lines:
            line = line.strip()
            if line:
                memory_id = self._generate_memory_id()
                theory_of_mind_items.append({
                    "memory_id": memory_id,
                    "mentioned_at": session_date,
                    "content": line,
                    "links": "",
                })
        
        return theory_of_mind_items
    
    def _generate_memory_id(self) -> str:
        """生成唯一的记忆ID"""
        return str(uuid.uuid4()).replace('-', '')[:8]