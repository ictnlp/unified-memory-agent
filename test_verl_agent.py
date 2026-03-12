"""VerlMemoryAgent 测试脚本"""
import os
os.environ['PROMPT_TEMPLATE_PATH'] = "./prompt_template.yaml"
os.environ['EMBEDDING_SERVICE_ENDPOINT'] = 'http://localhost:8080/embeddings'
import asyncio
from openai import AsyncOpenAI
from agents.verl_agent import VerlMemoryAgent

client = AsyncOpenAI(
    base_url='http://localhost:8000/v1',
    api_key='EMPTY'
)

async def test_simple():
    """简化测试"""
    print("=" * 80)
    print("VerlMemoryAgent 简化测试")
    print("=" * 80)

    agent = VerlMemoryAgent(
        client=client,
        model_name="dp66/UMA-4B",
        data_source="test_simple"
    )
    print(f"✓ Agent created\n")

    await agent.add_memory_async("""DATE: 2024-01-01
CONVERSATION:
User said, "My name is Alice. I work at Google in San Francisco. Please store this information with title 'personal_info'."
Assistant said, "Nice to meet you, Alice! San Francisco is a beautiful city. I will remember that."
""")
    await agent.add_memory_async("""DATE: 2024-01-02
CONVERSATION:
User said, "Please remember that my favorite color is blue. Use memory_add with title 'favorite_color' to store it."
Assistant said, "Got it! I'll remember that your favorite color is blue."
""")
    await agent.add_memory_async("""DATE: 2024-01-03
CONVERSATION:
User said, "Please use memory_update to change my favorite color to green."
Assistant said, "Got it! I'll remember that your favorite color is green."
""")
    await agent.add_memory_async("""DATE: 2024-01-04
CONVERSATION:
User said, "Please use memory_delete to delete my personal_info."
Assistant said, "Got it! I'll forget your personal_info."
""")
    print(f"✓ 已添加记忆（{len(agent._context_chunks)} 个片段）\n")

    questions = [
        "Where does Alice work?",
        "What is Alice's favorite color?",
        "What is the user's name? You MUST use the tool memory_embedding_retrieve to search for relevant information."
    ]
    print(f"Q: {questions}")

    results, intermediate_path, _ = await agent.QA_batch_async(questions, save_intermediate=True)
    print(f"A: {results}\n")
    print("✓ 测试成功!")
    if intermediate_path:
        print(f"✓ 中间过程: {intermediate_path}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_simple())
