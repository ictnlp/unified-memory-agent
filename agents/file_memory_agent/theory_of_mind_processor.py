"""
Step 2: Theory of Mind Processor
根据MemU的run_theory_of_mind实现
"""

import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Any


class TheoryOfMindProcessor:
    """处理心理理论分析的第二步：推理隐含信息"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def process(self, character_name: str, conversation_text: str, activity_items: List[Dict[str, str]], session_date: str = None) -> Dict[str, Any]:
        """
        执行步骤2：心理理论分析
        
        Args:
            character_name: 角色名称
            conversation_text: 完整对话文本
            activity_items: 活动记忆项列表
            session_date: 会话日期
            
        Returns:
            包含推理项的处理结果
        """
        try:
            if not conversation_text.strip():
                return {"success": False, "error": "Empty conversation text provided"}
            
            if not activity_items:
                return {"success": False, "error": "No memory items provided"}
            
            if not session_date:
                session_date = datetime.now().strftime("%Y-%m-%d")
            
            # 调用LLM进行心理理论分析
            response = self._extract_theory_of_mind_with_llm(
                character_name, conversation_text, activity_items
            )
            
            if not response.strip():
                return {"success": False, "error": "LLM returned empty response"}
            
            # 解析文本响应
            reasoning_process, theory_of_mind_items = self._parse_theory_of_mind_from_text(
                character_name, response.strip(), session_date
            )
            
            if not theory_of_mind_items:
                # 如果没有找到项目，返回成功但空项目而不是失败
                return {
                    "success": True,
                    "character_name": character_name,
                    "theory_of_mind_items_added": 0,
                    "theory_of_mind_items": [],
                    "reasoning_process": reasoning_process if reasoning_process else "No implicit information could be inferred from the conversation.",
                    "message": "No theory of mind items could be extracted from conversation",
                }
            
            return {
                "success": True,
                "character_name": character_name,
                "theory_of_mind_items_added": len(theory_of_mind_items),
                "theory_of_mind_items": theory_of_mind_items,
                "reasoning_process": reasoning_process,
                "message": f"Successfully extracted {len(theory_of_mind_items)} theory of mind items from conversation",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_theory_of_mind_with_llm(self, character_name: str, conversation_text: str, activity_items: List[Dict[str, str]]) -> str:
        """使用LLM从对话和活动项中提取心理理论项，完全按照MemU的原始prompt"""
        
        activity_items_text = "\n".join([
            f"- {item['content']}"
            for item in activity_items
        ])
        
        user_name = character_name
        
        # 完全按照MemU的原始prompt
        theory_of_mind_prompt = f"""You are analyzing the following conversation and activity items for {user_name} to try to infer information that is not explicitly mentioned by {user_name} in the conversation, but he or she might meant to express or the listener can reasonably deduce.

Conversation:
{conversation_text}

Activity Items:
{activity_items_text}

**CRITICAL REQUIREMENT: Inference results must be SELF-CONTAINED MEMORY ITEMS**

Your task it to leverage your reasoning skills to infer the information that is not explicitly mentioned in the conversation, but the character might meant to express or the listener can reasonably deduce.

**SELF-CONTAINED MEMORY REQUIREMENTS:**
- Plain text only, no markdown grammar
- EVERY activity item must be complete and standalone
- ALWAYS include the full subject (do not use "she/he/they/it")
- NEVER use pronouns that depend on context (no "she", "he", "they", "it")
- Include specific names, places, dates, and full context in each item
- Each activity should be understandable without reading other items
- You can use words like "perhaps" or "maybe" to indicate that the information is obtained through reasoning and is not 100% certain
- NO need to include evidences or reasoning processes in the items

**INFERENCE GUIDELINES:**
- Leverage your reasoning skills to infer the information that is not explicitly mentioned
- Use the activity items as a reference to assist your reasoning process and inferences
- DO NOT repeat the information that is already included in the activity items
- Use modal adverbs (perhaps, probably, likely, etc.) to indicate your confidence level of the inference

**COMPLETE SENTENCE EXAMPLES:**
GOOD: "{user_name} may have experience working abroad"
BAD: "Have experience working abroad" (missing subject)
BAD: "He may have experience working abroad" (pronouns as subject)
GOOD: "{user_name} perhaps not enjoy his trip to Europe this summer"
BAD: "{user_name} perhaps not enjoy his trip" (missing location and time)
GOOD: "Harry Potter series are probably important to {user_name}'s childhood"
BAD: "Harry Potter series are probably important to {user_name}'s childhood, because she mentioned it and recommended it to her friends many times" (no need to include evidences or reasoning processes)

**OUTPUT FORMAT:**

**REASONING PROCESS:**
[Your reasoning process for what kind of implicit information can be hidden behind the conversation, what are the evidences, how you get to your conclusion, and how confident you are.]

**INFERENCE ITEMS:**
[One piece of inference per line, no markdown headers, no structure, no numbering, no bullet points, ends with a period]
[After carefully reasoning, if you determine that there is no implicit information that can be inferred from the conversation beyong the explicit information already mentioned in the activity items, you can leave this section empty. DO NOT output things like "No inference available".]

"""
        
        # 调用LLM进行心理理论分析
        response = self.llm_client.chat.completions.create(
            model="azure-gpt-4_1",
            messages=[{"role": "user", "content": theory_of_mind_prompt}],
            max_tokens=2048,
            temperature=0.1
        ).choices[0].message.content
        
        return response
    
    def _parse_theory_of_mind_from_text(self, character_name: str, response_text: str, session_date: str) -> Tuple[str, List[Dict]]:
        """从文本格式响应中解析心理理论项"""
        
        reasoning_process = ""
        theory_of_mind_items = []
        
        try:
            lines = response_text.split("\n")
            
            # 解析推理过程
            reasoning_section = False
            inference_section = False
            
            for line in lines:
                line = line.strip()
                
                if (line.upper().startswith("**REASONING PROCESS:") or 
                    line.startswith("**") and "REASONING PROCESS" in line.upper()):
                    reasoning_section = True
                    inference_section = False
                    continue
                elif (line.upper().startswith("**INFERENCE ITEMS:") or 
                      line.startswith("**") and "INFERENCE ITEMS" in line.upper()):
                    reasoning_section = False
                    inference_section = True
                    continue
                
                if reasoning_section and line and not line.startswith("**"):
                    if not reasoning_process:
                        reasoning_process = line.strip()
                    else:
                        reasoning_process += "\n" + line.strip()
                
                # 解析记忆项
                elif inference_section:
                    line = line.strip()
                    if line:
                        memory_id = self._generate_memory_id()
                        theory_of_mind_items.append({
                            "memory_id": memory_id,
                            "mentioned_at": session_date,
                            "content": line,
                            "links": "",
                        })
                        
        except Exception as e:
            print(f"Failed to parse theory of mind from text: {repr(e)}")
        
        return reasoning_process, theory_of_mind_items
    
    def _generate_memory_id(self) -> str:
        """生成唯一的记忆ID"""
        return str(uuid.uuid4()).replace('-', '')[:8]