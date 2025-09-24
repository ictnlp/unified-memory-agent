#!/usr/bin/env python3
"""
测试优化后的FileMemoryAgent - 验证合并的CombinedProcessor
"""

import sys
import os
sys.path.append('/mnt/pfs-guan-ssai/nlu/zhangkehao/Unified_Memory_Agent')

from agents.file_memory_agent_dev.combined_processor import CombinedProcessor
from openai import OpenAI
from config import API_CONFIG

def test_combined_processor():
    """测试合并的处理器功能"""
    
    # 创建客户端
    client = OpenAI(**API_CONFIG)
    
    # 创建合并处理器
    processor = CombinedProcessor(client)
    
    # 测试对话
    test_conversation = """用户: 我今天去了一个LGBTQ支持小组的聚会。
助手: 那听起来很有意义，你感觉怎么样？
用户: 很好！我听到了一些跨性别者的励志故事，让我感到很开心和感激。我觉得被接纳了，也更有勇气做真实的自己。
助手: 这真是太棒了。
用户: 是的，而且我还和Melanie讨论了未来的职业规划。我对咨询和心理健康工作很感兴趣，想要帮助有类似问题的人。
助手: 听起来你很有同理心。
用户: Melanie说我会是个很好的咨询师，因为我很有同理心和理解力。另外，我还欣赏了Melanie去年画的湖边日出，颜色搭配得很棒。
助手: 绘画是很好的表达方式吗？
用户: 是的，我们聊到绘画是表达情感和在忙碌后放松的好方式。"""
    
    print("🧪 Testing Combined Processor...")
    print(f"Input conversation length: {len(test_conversation)} characters")
    
    # 调用合并处理器
    result = processor.process(
        character_name="用户",
        content=test_conversation,
        session_date="2023-05-20"
    )
    
    print(f"\n📊 Processing Results:")
    print(f"Success: {result.get('success')}")
    
    if result.get('success'):
        activity_items = result.get('activity_items', [])
        theory_items = result.get('theory_of_mind_items', [])
        
        print(f"Activity items generated: {len(activity_items)}")
        print(f"Theory of mind items generated: {len(theory_items)}")
        print(f"Total memory items: {len(result.get('memory_items', []))}")
        
        print(f"\n📝 Activity Memory Items:")
        for i, item in enumerate(activity_items, 1):
            print(f"{i}. {item['content']}")
        
        print(f"\n🧠 Theory of Mind Items:")
        for i, item in enumerate(theory_items, 1):
            print(f"{i}. {item['content']}")
        
        print(f"\n🤔 Reasoning Process:")
        print(result.get('reasoning_process', 'No reasoning provided'))
        
        print(f"\n✅ Combined processor test completed successfully!")
        print(f"Efficiency gain: Reduced from 2 LLM calls to 1 LLM call")
        
    else:
        print(f"❌ Processing failed: {result.get('error')}")

if __name__ == "__main__":
    test_combined_processor()