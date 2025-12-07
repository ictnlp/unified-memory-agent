# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class TrajectoryConversation:
    """Represents a single conversation within a trajectory."""

    def __init__(self, prompt_ids: list[int], response_ids: list[int], response_mask: list[int],
                 response_logprobs: list[float], is_final: bool = False, user_turns: int = 0, assistant_turns: int = 0, tool_rewards: list[float] = [], num_tools: int = 0):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_mask = response_mask
        self.response_logprobs = response_logprobs
        self.is_final = is_final
        self.user_turns = user_turns
        self.assistant_turns = assistant_turns
        self.tool_rewards = tool_rewards
        self.num_tools = num_tools


class AgentData:
    """Encapsulates all state variables for the memory agent loop with tool support."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        memory_kwargs: dict[str, Any],
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        self.memory_kwargs = memory_kwargs
        self.tools_kwargs = tools_kwargs

        # Trajectory tracking
        self.conversations: list[TrajectoryConversation] = []
        
        # State variables for current conversation (accumulative within a conversation)
        self.current_prompt_ids: list[int] = []
        self.current_response_ids: list[int] = []
        self.current_response_mask: list[int] = []
        self.current_response_logprobs: list[float] = []
        self.current_num_tools: int = 0
        
        # Memory management
        self.memory_turns = 0
        self.memory_content = memory_kwargs.get("initial_memory", "")
        self.context_chunks = memory_kwargs.get("context_chunks", [])
        self.current_chunk_idx = 0
        
        # Tool-related state
        self.tool_calls: list[FunctionCall] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.current_tool_rewards: list[float] = []
    
    def add_conversation(self, is_final: bool = False):
        """Add a conversation to the trajectory using current accumulated state."""
        response_token_count = len(self.current_response_mask)
        prompt_token_count = max(len(self.current_prompt_ids) - response_token_count, 0)
        prompt_ids = self.current_prompt_ids[:prompt_token_count]
        conversation = TrajectoryConversation(
            prompt_ids=prompt_ids,
            response_ids=self.current_response_ids.copy(),
            response_mask=self.current_response_mask.copy(),
            response_logprobs=self.current_response_logprobs.copy(),
            is_final=is_final,
            user_turns=self.user_turns,
            assistant_turns=self.assistant_turns,
            tool_rewards=self.current_tool_rewards,
            num_tools=self.current_num_tools,
        )
        self.conversations.append(conversation)
        
        # Reset current state for next conversation
        self.current_prompt_ids = []
        self.current_response_ids = []
        self.current_response_mask = []
        self.current_response_logprobs = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.current_tool_rewards = []
        self.current_num_tools = 0

def get_chunks(context_text, chunk_size):
    separator = "\n"
    small_chunks = context_text.split(separator)
    chunks = []
    to_append = ""
    for chunk in small_chunks:
        if len(to_append) + len(chunk) + len(separator) <= chunk_size:
            if to_append:
                to_append += separator + chunk
            else:
                to_append = chunk
        else:
            if to_append:
                chunks.append(to_append)
            to_append = chunk
    if to_append:
        chunks.append(to_append)
    return chunks

@register("tool_mem_agent")
class ToolMemoryAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolMemoryAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        
        # Memory-specific configuration
        cls.max_chunk_size = config.actor_rollout_ref.rollout.multi_turn.get("max_chunk_size", 15000)
        
        # Tool-specific configuration
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.get("max_parallel_calls", 5)
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.get("max_tool_response_length", 1000)
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.get("tool_response_truncate_side", "right")
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.get("tool_config_path", None)
        
        if tool_config_path:
            tool_list = initialize_tools_from_config(tool_config_path)
            cls.tools = {tool.name: tool for tool in tool_list}
            cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
            cls.tool_parser = ToolParser.get_tool_parser(
                config.actor_rollout_ref.rollout.multi_turn.get("format", "default"), 
                cls.tokenizer
            )
            print(f"Initialized tools: {list(cls.tools.keys())}")
        else:
            cls.tools = {}
            cls.tool_schemas = []
            cls.tool_parser = None
            print("No tools configured")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        # print("###################DEBUG######################")
        # print("KWARGS for ToolMemoryAgentLoop run:", kwargs.keys())
        # print("tools_kwargs:", kwargs.get("tools_kwargs", {}).keys())
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        memory_kwargs = kwargs.get("memory_kwargs", {})
        tools_kwargs = kwargs.get("tools_kwargs", {})
        # Initialize context chunks from the prompt
        context_text = kwargs.get("context", "")
        if context_text:
            # Split context into chunks
            chunk_size = memory_kwargs.get("chunk_size", self.max_chunk_size)
            context_chunks = get_chunks(context_text, chunk_size)
            memory_kwargs["context_chunks"] = context_chunks
        else:
            memory_kwargs["context_chunks"] = []

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            memory_kwargs=memory_kwargs,
            tools_kwargs=tools_kwargs,
        )
        trajectory_id = str(uuid4())
        # Process each chunk as a separate conversation with multi-turn tool calling
        for chunk_idx in range(len(agent_data.context_chunks)):
            agent_data.current_chunk_idx = chunk_idx
            await self._process_memory_chunk(agent_data, sampling_params, kwargs.get('verbose'), trajectory_id)
        # Generate final response with tool support
        await self._process_final_response(agent_data, sampling_params, kwargs.get('verbose'), trajectory_id)

        # Create a list of AgentLoopOutput objects, one for each conversation
        outputs = []
        first_final_position = -1
        for i, conversation in enumerate(agent_data.conversations):
            is_final = conversation.is_final
            if is_final and first_final_position == -1:
                first_final_position = i
            output = AgentLoopOutput(
                prompt_ids=conversation.prompt_ids,
                response_ids=conversation.response_ids[: self.response_length],
                response_mask=conversation.response_mask[: self.response_length],
                multi_modal_data={},
                response_logprobs=conversation.response_logprobs[: self.response_length]
                if conversation.response_logprobs
                else None,
                num_turns=conversation.user_turns + conversation.assistant_turns + 1,
                metrics=agent_data.metrics,
                extra_fields={'trajectory_id': trajectory_id, 'is_final': is_final, 'tool_rewards': conversation.tool_rewards, 
                              'num_tools': conversation.num_tools, 'query_id': i - first_final_position},
            )
            outputs.append(output)
        
        return outputs

    async def _process_memory_chunk(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool, trajectory_id: str):
        """Process a single memory chunk with multi-turn tool calling."""
        # Prepare memory update prompt
        current_chunk = agent_data.context_chunks[agent_data.current_chunk_idx]
        memory_prompt = self._build_memory_prompt(
            agent_data.messages, agent_data.memory_content, current_chunk
        )
        
        # Initialize prompt for this conversation
        agent_data.current_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": memory_prompt}],
                tools=self.tool_schemas if self.tools else None,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        if len(agent_data.current_prompt_ids) > self.prompt_length + self.response_length:
            print(f"WARNING: Memory prompt length {len(agent_data.current_prompt_ids)} exceeds max prompt length {self.prompt_length + self.response_length}")
        # Multi-turn generation loop for this chunk
        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, trajectory_id)
            
            # Check termination conditions for this conversation
            if len(agent_data.current_response_mask) >= self.response_length:
                state = AgentState.TERMINATED
            if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                state = AgentState.TERMINATED

        await self._update_memory_content(agent_data, verbose, trajectory_id)
        agent_data.add_conversation(is_final=False)
        agent_data.memory_turns += 1

    async def _process_final_response(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool, trajectory_id: str):
        """Generate final response with tool support."""
        # Prepare final prompt
        final_prompts = self._build_final_prompt(agent_data.messages, agent_data.memory_content)
        if isinstance(final_prompts, str):
            final_prompts = [final_prompts]
        for final_prompt in final_prompts:
            agent_data.current_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": final_prompt}],
                    tools=self.tool_schemas if self.tools else None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if len(agent_data.current_prompt_ids) > self.prompt_length + self.response_length:
                print(f"WARNING: Final prompt length {len(agent_data.current_prompt_ids)} exceeds max prompt length {self.prompt_length + self.response_length}")
            # Multi-turn generation loop for final response
            state = AgentState.GENERATING
            while state != AgentState.TERMINATED:
                if state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data, trajectory_id)
                
                # Check termination conditions
                if len(agent_data.current_response_mask) >= self.response_length:
                    state = AgentState.TERMINATED
                if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                    state = AgentState.TERMINATED
            
            if verbose:
                response_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(agent_data.current_response_ids, skip_special_tokens=True),
                )
                print(f"Final Answer:\n{response_text}")
            
            agent_data.add_conversation(is_final=True)

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        max_model_len = self.prompt_length + self.response_length
        max_new_tokens = min(self.response_length, max_model_len - len(agent_data.current_prompt_ids) - 1)
        if max_new_tokens <= 0:
            print(f"ERROR: Current len(agent_data.current_prompt_ids)={len(agent_data.current_prompt_ids)} exceeds model max length {max_model_len}")
        with simple_timer("generate_sequences", agent_data.metrics):
            # 走的async_sglang_server.py
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.current_prompt_ids,
                sampling_params=sampling_params,
            )
        agent_data.assistant_turns += 1
        response_ids = output.token_ids
        # if "<tool_call>" in self.tokenizer.decode(response_ids, skip_special_tokens=True):
        #     breakpoint()
        agent_data.current_prompt_ids += response_ids
        agent_data.current_response_ids += response_ids
        agent_data.current_response_mask += [1] * len(response_ids)
        if output.log_probs:
            agent_data.current_response_logprobs += output.log_probs
        else:
            agent_data.current_response_logprobs += [0.0] * len(response_ids)

        # Extract tool calls if tool parser is available
        agent_data.tool_calls = []
        if self.tool_parser:
            _, agent_data.tool_calls, num_tools = await self.tool_parser.extract_tool_calls(response_ids, return_num_tools=True)
            agent_data.current_num_tools += num_tools
        
        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData, session_id: str) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, session_id=session_id))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses
        for tool_response, tool_reward, _ in responses:
            message = {"role": "tool", "content": tool_response.text or ""}
            add_messages.append(message)
            if tool_reward is not None:
                agent_data.current_tool_rewards.append(tool_reward)
        # Update prompt with tool responses
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, 
                add_generation_prompt=True, 
                tokenize=True,
                **self.apply_chat_template_kwargs
            ),
        )
        response_ids = response_ids[len(self.system_prompt):]
        
        # Check if we would exceed response length
        if len(agent_data.current_response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        if len(agent_data.current_prompt_ids) + len(response_ids) >= self.prompt_length + self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.current_prompt_ids += response_ids
        agent_data.current_response_ids += response_ids
        agent_data.current_response_mask += [0] * len(response_ids)
        agent_data.current_response_logprobs += [0.0] * len(response_ids)
        
        # Clear tool calls for next round
        agent_data.tool_calls = []
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _update_memory_content(self, agent_data: AgentData, verbose: bool, trajectory_id: str):
        """Update memory content from the generated response."""
        # Decode the response to get updated memory
        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.current_response_ids, skip_special_tokens=True),
        )

        # Extract memory content (remove thinking tags if present)
        for sep in ['</tool_response>\nassistant', '</think>']:
            if sep in response_text:
                response_text = response_text.split(sep)[-1].strip()

        if verbose:
            print(f"Updated memory:\n{response_text}")
            print(f"TRAJECTORY_ID: {trajectory_id}")
            # print(f"Memory turn: {agent_data.memory_turns}")
            print(f"Current chunk index: {agent_data.current_chunk_idx + 1} / {len(agent_data.context_chunks)}")
            current_chunk_idx = agent_data.current_chunk_idx
            current_full_chunk = agent_data.context_chunks[current_chunk_idx]
            show_len = 100
            if len(current_full_chunk) <= show_len:
                print(f"Current chunk:\n{current_full_chunk}")
            else:
                print(fr"Current chunk (truncated):\n{current_full_chunk[:show_len // 2]}...(truncated)...{current_full_chunk[-show_len // 2:]}")
        
        agent_data.memory_content = response_text

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], session_id: str) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            create_kwargs = kwargs.get("create_kwargs", {})
            create_kwargs['session_id'] = session_id
            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing {tool_name} tool: {e}")
            return ToolResponse(text=f"Error when executing {tool_name} tool: {e}"), 0.0, {}
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        return ToolResponse(text=tool_response_text), tool_reward, res

    def _build_memory_prompt(self, messages: list[dict[str, Any]], memory: str, chunk: str) -> str:
        """Build prompt for memory update."""
#         memory_prompt = f"""You are an expert assistant who can remember key information by using tool calls. You are presented with a section of context that may contain relevant information and a previous memory.

# Please read the provided section carefully and update the memory with the new information that helps to answer the problem. 
# Be sure to retain all relevant details from the previous memory while adding any new, useful information.

# In addition, you need to use tools to help you update the memory. You can use the tools to add new information, remove irrelevant details, or modify existing entries in the memory. When the key information appears in the context, you **MUST** use the memory_add tool to add the memory accordingly.

# To provide the updated memory, answer directly in the text without calling any tools. It is the only way to complete the task, else you will be stuck on a loop. When you do not need to call any tools, just continue generating the updated memory until the end.

# <memory>
# {memory if memory else "No previous memory"}
# </memory>

# <section>
# {chunk}
# </section>

# Updated memory:
        memory_prompt = f"""
You are an expert assistant for memory management.

### Step 1: Core Memory Update
- Core memory should store only high-level themes and main topics of the context.
- Do NOT include detailed facts, numerical data, timestamps, or full event descriptions.
- Summarize in 1–3 short sentences what the context is about:
  - Main subject / theme
  - Overall purpose or intent
  - Any broad category or domain (e.g., technology, sports, finance, etc.)
- Keep it concise, clear, and relevant to understanding "what this context is generally about".
- If the new section changes or adds themes, update the core memory accordingly.
- Avoid redundancy; merge overlapping topics.

### Step 2: External Memory Storage (Tool Calls)
- **Episodic Memory**: record specific events, user actions, friend actions, or assistant actions with timestamps.
- **Semantic Memory**: record detailed facts, definitions, relationships, labels.
- Always store these details externally via tools like `memory_add`.
- **Do NOT store the Core Memory externally** — it is for internal topic reference only.

<core_memory>
{memory if memory else "No previous core memory"}
</core_memory>

<section>
{chunk}
</section>

**Important**:
- Final answer must be the updated concise core memory only (plain text, no tool syntax).
- Max length: 1–3 sentences.
- Focus only on WHAT the context is generally about, not details.

Updated concise core memory:
"""
        return memory_prompt

    def _build_final_prompt(self, messages: list[dict[str, Any]], memory: str) -> str:
        """Build prompt for final response."""
        prompt_text = messages[0]["content"] if messages[0]["role"] == "user" else messages[1]["content"]
        if isinstance(prompt_text, str):
            final_prompt = f"""You are presented with a problem and a previous memory.

You have access to tools that can help you solve the problem. Use them if necessary. In particular, if you can't find the answer in the memory, you should use the **memory_list**, **memory_key_retrieve**, **memory_embedding_retrieve** and **memory_bm25_retrieve** tools to find relevant information. If you can not find the answer using one of these tools, you can also use other tools to help you. Try as hard as you can to find the answer using the tools available to you until you find the answer or you are sure that the answer is not in the memory or tools. Do not decide that the answer is not in the memory or tools unless you have used all relevant tools to search for the answer. Before you decided that the information is not available, check that you have used all relevant tools to search for the answer.

Please answer the problem based on the previous memory and the tools. Put your final answer in \\boxed{{}}.

<problem>
{prompt_text}
</problem>

<memory>
{memory if memory else "No previous memory"}
</memory>

Your answer:"""
        elif isinstance(prompt_text, list):
            final_prompt = [f"""You are presented with a problem and a previous memory.

You have access to tools that can help you solve the problem. Use them if necessary. In particular, if you can't find the answer in the memory, you should use the **memory_list**, **memory_key_retrieve**, **memory_embedding_retrieve** and **memory_bm25_retrieve** tools to find relevant information. If you can not find the answer using one of these tools, you can also use other tools to help you. Try as hard as you can to find the answer using the tools available to you until you find the answer or you are sure that the answer is not in the memory or tools. Do not decide that the answer is not in the memory or tools unless you have used all relevant tools to search for the answer. Before you decided that the information is not available, check that you have used all relevant tools to search for the answer.

Please answer the problem based on the previous memory and the tools. Put your final answer in \\boxed{{}}.

<problem>
{p}
</problem>

<memory>
{memory if memory else "No previous memory"}
</memory>

Your answer:""" for p in prompt_text]
        return final_prompt