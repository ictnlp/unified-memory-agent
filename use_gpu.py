import asyncio
from argparse import ArgumentParser
from openai import AsyncOpenAI

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="YuWangX/Memalpha-4B")
parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
args = parser.parse_args()

client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")


async def make_request(semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model=args.model,
            # model="Qwen/Qwen3-4B-Instruct-2507",
            # model="/lpai/volumes/base-agentos-lx-my/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_data/hf",
            # model="YuWangX/Memalpha-4B",
        )
        print(response.choices[0].message.content)


async def main() -> None:
    semaphore = asyncio.Semaphore(500)
    pending: set[asyncio.Task[None]] = set()

    def _cleanup(task: asyncio.Task[None]) -> None:
        pending.discard(task)
        if exc := task.exception():
            print(f"Task failed: {exc}")

    try:
        while True:
            if len(pending) >= 500:
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for finished in done:
                    _cleanup(finished)

            task = asyncio.create_task(make_request(semaphore))
            task.add_done_callback(_cleanup)
            pending.add(task)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())