"""
Step 1: Activity Memory Processor
根据MemU的add_activity_memory实现
"""

import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Any


class ActivityProcessor:
    """处理活动记忆的第一步：格式化对话为自包含的记忆项"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def process(self, character_name: str, content: str, session_date: str = None) -> Dict[str, Any]:
        """
        执行步骤1：添加活动记忆
        
        Args:
            character_name: 角色名称
            content: 原始对话内容
            session_date: 会话日期
            
        Returns:
            处理结果包含格式化的记忆项
        """
        try:
            if not session_date:
                session_date = datetime.now().strftime("%Y-%m-%d")
            
            # 使用LLM格式化内容
            formatted_content = self._format_content_with_llm(character_name, content, session_date)
            
            if not formatted_content.strip():
                return {"success": False, "error": "LLM returned empty formatted content"}
            
            # 添加记忆ID和时间戳
            memory_items, content_with_ids = self._add_memory_ids_with_timestamp(
                formatted_content, session_date
            )
            
            return {
                "success": True,
                "character_name": character_name,
                "category": "activity",
                "session_date": session_date,
                "memory_items_added": len(memory_items),
                "memory_items": memory_items,
                "formatted_content": content_with_ids,
                "message": f"Successfully generated {len(memory_items)} self-contained activity memory items for {character_name}",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _format_content_with_llm(self, character_name: str, content: str, session_date: str) -> str:
        """使用LLM格式化内容，完全按照MemU的原始prompt"""
        
        user_name = character_name
        
        # 完全按照MemU的原始prompt
        format_prompt = f"""You are formatting activity memory content for {user_name} on {session_date}.

Raw content to format:
{content}

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

**GOOD EXAMPLES (meaningful activities, one per line):**
{character_name} attended a LGBTQ support group where {character_name} heard inspiring transgender stories and felt happy, thankful, accepted, and gained courage to embrace {character_name}'s true self.
{character_name} discussed future career plans with Melanie, expressing keen interest in counseling and mental health work to support people with similar issues, and Melanie encouraged {character_name} saying {character_name} would be a great counselor due to {character_name}'s empathy and understanding.
{character_name} admired Melanie's lake sunrise painting from last year, complimented the color blending, and discussed how painting serves as a great outlet for expressing feelings and relaxing after long days.

**BAD EXAMPLES (too fragmented):**
{character_name} went to a LGBTQ support group.
{character_name} heard transgender stories.
{character_name} felt happy and thankful.
{character_name} gained courage to embrace {character_name}'s true self.

**ACTIVITY GROUPING GUIDELINES:**
- Conversations about the same topic → Single activity
- Related actions and their outcomes → Single activity
- Emotional reactions to specific events → Include in the main activity
- Sequential related events → Single comprehensive activity
- Different topics or unrelated events → Separate activities

**QUALITY STANDARDS:**
- Never use "he", "she", "they", "it" - always use the person's actual name
- Never use "the book", "the place", "the friend" - always include full titles and names
- Each activity must be complete and tell the full story
- Include emotional context, outcomes, and significance
- Merge related content intelligently to create meaningful activity summaries

Transform the raw content into properly formatted activity memory items (ONE MEANINGFUL ACTIVITY PER LINE):

"""
        
        # 调用LLM格式化内容
        cleaned_content = self.llm_client.chat.completions.create(
            model="azure-gpt-4_1",
            messages=[{"role": "user", "content": format_prompt}],
            max_tokens=2048,
            temperature=0.1
        ).choices[0].message.content
        
        return cleaned_content
    
    def _add_memory_ids_with_timestamp(self, content: str, session_date: str) -> Tuple[List[Dict], str]:
        """
        为内容行添加记忆ID和时间戳
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
    
    def _generate_memory_id(self) -> str:
        """生成唯一的记忆ID"""
        return str(uuid.uuid4()).replace('-', '')[:8]