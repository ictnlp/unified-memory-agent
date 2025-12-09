"""
Memory Agent Tool for SGLang Multi-turn Tool Calling

This module provides memory management capabilities through SGLang's native tool calling interface.
It integrates with the recurrent memory agent system to provide persistent memory across turns.
"""

import asyncio
import json
import logging
import os
import string
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import util

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema
from verl.utils.rollout_trace import rollout_trace_op

from ..schemas import OpenAIFunctionToolSchema, OpenAIFunctionSchema, OpenAIFunctionParametersSchema, ToolResponse

logger = logging.getLogger(__name__)


class AddTool(BaseTool):
    """Add new memory entries identified by a human-readable title."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionCallSchema:
        """Return the OpenAI-compatible schema for this tool."""
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_add",
                description="Store new information in memory with a unique title for later retrieval",
                parameters=OpenAIFunctionParametersSchema(**{
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Human-friendly title used to reference this memory entry"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to store in memory"
                        }
                    },
                    "required": ["title", "content"]
                })
            )
        )
    
    async def create(self, **kwargs) -> str:
        """Create a new memory session."""
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))
        filename = kwargs.get("create_kwargs", {}).get("filename", "memory_store.jsonl")
        filename = filename[:-6] + f"_{trajectory_id}.jsonl"
        self._instance_dict[trajectory_id] = {
            "filename": filename
        }
        return trajectory_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        arguments: Dict[str, Any], 
        **kwargs
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Execute the memory operation.
        
        Args:
            instance_id: Session ID for this memory instance
            arguments: Tool arguments from the model
            **kwargs: Additional execution parameters
            
        Returns:
            Tuple of (response, reward, metrics)
        """
        title = arguments.get("title")
        content = arguments.get("content")
        filename = self._instance_dict[instance_id]["filename"]
        if title is None or content is None:
            return ToolResponse(text="Missing title or content"), 0, {"error": "missing_title_or_content"}
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open(filename, "w").close()
        existing_memories = [json.loads(line) for line in open(filename, "r")]

        existing_titles = [mem.get("title") or mem.get("key") for mem in existing_memories]
        if title in existing_titles:
            response = f"Title '{title}' already exists. Please use a different title or the memory_update tool."
            reward = 0
            metrics = {}
        else:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")
            response = f"Successfully stored content under title '{title}'"
            reward = 1  # Small positive reward for successful storage
            metrics = {}
        return ToolResponse(text=response), reward, metrics

    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class UpdateTool(BaseTool):
    """Update an existing memory entry by title, optionally renaming it."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionCallSchema:
        """Return the OpenAI-compatible schema for this tool."""
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_update",
                description="Update or rename existing memory entries by title, replacing their content completely",
                parameters=OpenAIFunctionParametersSchema(**{
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Existing memory title to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content to persist"
                        },
                        "new_title": {
                            "type": "string",
                            "description": "Optional new title for the memory entry"
                        }
                    },
                    "required": ["title", "content"]
                })
            )
        )
    
    async def create(self, **kwargs) -> str:
        """Create a new memory session."""
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))
        filename = kwargs.get("create_kwargs", {}).get("filename", "memory_store.jsonl")
        filename = filename[:-6] + f"_{trajectory_id}.jsonl"
        self._instance_dict[trajectory_id] = {
            "filename": filename
        }
        return trajectory_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        arguments: Dict[str, Any], 
        **kwargs
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Execute the memory update operation.
        
        Args:
            instance_id: Session ID for this memory instance
            arguments: Tool arguments from the model
            **kwargs: Additional execution parameters
            
        Returns:
            Tuple of (response, reward, metrics)
        """
        title = arguments.get("title")
        content = arguments.get("content")
        new_title = arguments.get("new_title")
        filename = self._instance_dict[instance_id]["filename"]
        if title is None or content is None:
            return ToolResponse(text="Missing title or content"), 0, {"error": "missing_title_or_content"}

        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open(filename, "w").close()
        existing_memories = [json.loads(line) for line in open(filename, "r")]

        titles = [mem["title"] for mem in existing_memories]
        if title not in titles:
            response = f"Title '{title}' does not exist. Please use the memory_add tool to create it first."
            reward = 0
            metrics = {}
        else:
            target_index = titles.index(title)
            if new_title and new_title != title and new_title in titles:
                response = f"Title '{new_title}' already exists. Please choose a different title."
                reward = 0
                metrics = {}
            else:
                updated_memories = []
                for idx, mem in enumerate(existing_memories):
                    if idx == target_index:
                        updated_memories.append(
                            {
                                "title": new_title or title,
                                "content": content,
                            }
                        )
                    else:
                        other_title = mem["title"] or "(untitled)"
                        updated_memories.append(
                            {
                                "title": other_title,
                                "content": mem.get("content", ""),
                            }
                        )
                with open(filename, "w", encoding="utf-8") as f:
                    for mem in updated_memories:
                        f.write(json.dumps(mem, ensure_ascii=False) + "\n")
                original_entry = existing_memories[target_index]
                original_title = original_entry["title"] or "(untitled)"
                original_content = original_entry.get("content", "")
                effective_title = new_title or original_title
                response = (
                    f"Successfully updated entry titled '{original_title}'. "
                    f"New title: '{effective_title}', New content: '{content}'."
                )
                reward = 1  # Small positive reward for successful update
                metrics = {}
        return ToolResponse(text=response), reward, metrics
    
    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class DeleteTool(BaseTool):
    """Delete a stored memory entry using its title."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionCallSchema:
        """Return the OpenAI-compatible schema for this tool."""
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_delete",
                description="Permanently delete memory entries by their exact title",
                parameters=OpenAIFunctionParametersSchema(**{
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Memory title to delete"
                        }
                    },
                    "required": ["title"]
                })
            )
        )
    
    async def create(self, **kwargs) -> str:
        """Create a new memory session."""
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))
        filename = kwargs.get("create_kwargs", {}).get("filename", "memory_store.jsonl")
        filename = filename[:-6] + f"_{trajectory_id}.jsonl"
        self._instance_dict[trajectory_id] = {
            "filename": filename
        }
        return trajectory_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        arguments: Dict[str, Any], 
        **kwargs
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Execute the memory deletion operation.
        
        Args:
            instance_id: Session ID for this memory instance
            arguments: Tool arguments from the model
            **kwargs: Additional execution parameters
            
        Returns:
            Tuple of (response, reward, metrics)
        """
        title = arguments.get("title")
        filename = self._instance_dict[instance_id]["filename"]
        if title is None:
            return ToolResponse(text="Missing title"), 0, {"error": "missing_title"}

        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open(filename, "w").close()
        existing_memories = [json.loads(line) for line in open(filename, "r")]

        titles = [mem['title'] for mem in existing_memories]
        if title not in titles:
            response = f"Title '{title}' does not exist."
            reward = 0
            metrics = {}
        else:
            updated_memories = [
                {"title": mem['title'] or "(untitled)", "content": mem.get("content", "")}
                for mem in existing_memories
                if mem['title'] != title
            ]
            with open(filename, "w", encoding="utf-8") as f:
                for mem in updated_memories:
                    f.write(json.dumps(mem, ensure_ascii=False) + "\n")
            response = f"Successfully deleted memory entry titled '{title}'."
            reward = 1  # Small positive reward for successful deletion
            metrics = {}
        return ToolResponse(text=response), reward, metrics
    
    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class RetrieveTool(BaseTool):
    """Retrieve the content of a memory entry by title."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionCallSchema:
        """Return the OpenAI-compatible schema for this tool."""
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_key_retrieve",
                description="Retrieve the full content of a memory entry using its exact title",
                parameters=OpenAIFunctionParametersSchema(**{
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Memory title to retrieve"
                        }
                    },
                    "required": ["title"]
                })
            )
        )
    
    async def create(self, **kwargs) -> str:
        """Create a new memory session."""
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))
        filename = kwargs.get("create_kwargs", {}).get("filename", "memory_store.jsonl")
        filename = filename[:-6] + f"_{trajectory_id}.jsonl"
        self._instance_dict[trajectory_id] = {
            "filename": filename
        }
        return trajectory_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        arguments: Dict[str, Any], 
        **kwargs
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Execute the memory retrieval operation.
        
        Args:
            instance_id: Session ID for this memory instance
            arguments: Tool arguments from the model
            **kwargs: Additional execution parameters
            
        Returns:
            Tuple of (response, reward, metrics)
        """
        title = arguments.get("title")
        filename = self._instance_dict[instance_id]["filename"]
        if title is None:
            return ToolResponse(text="Missing title"), 0, {"error": "missing_title"}

        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open(filename, "w").close()
        existing_memories = [json.loads(line) for line in open(filename, "r")]

        titles = [mem["title"] for mem in existing_memories]
        if title not in titles:
            response = f"Title '{title}' does not exist. Use memory_list to see available titles."
            reward = 0
            metrics = {}
        else:
            matched_index = titles.index(title)
            matched = existing_memories[matched_index]
            content = matched.get("content", "")
            response_lines = [f"Content for title '{title}': '{content}'"]
            response = "\n".join(response_lines)
            reward = 1  # Small positive reward for successful retrieval
            metrics = {}
        return ToolResponse(text=response), reward, metrics
    
    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class ListTool(BaseTool):
    """List all stored memory titles with short content previews."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionCallSchema:
        """Return the OpenAI-compatible schema for this tool."""
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_list",
                description="List all stored memory titles with content previews to see what's available",
                parameters=OpenAIFunctionParametersSchema(**{
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            )
        )
    
    async def create(self, **kwargs) -> str:
        """Create a new memory session."""
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))
        filename = kwargs.get("create_kwargs", {}).get("filename", "memory_store.jsonl")
        filename = filename[:-6] + f"_{trajectory_id}.jsonl"
        self._instance_dict[trajectory_id] = {
            "filename": filename
        }
        return trajectory_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        arguments: Dict[str, Any], 
        **kwargs
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Execute the memory listing operation.
        
        Args:
            instance_id: Session ID for this memory instance
            arguments: Tool arguments from the model
            **kwargs: Additional execution parameters
            
        Returns:
            Tuple of (response, reward, metrics)
        """
        filename = self._instance_dict[instance_id]["filename"]
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open(filename, "w").close()
        existing_memories = [json.loads(line) for line in open(filename, "r")]

        normalized_memories = []
        for mem in existing_memories:
            title = mem.get("title") or mem.get("key") or "(untitled)"
            normalized_memories.append({"title": title, "content": mem.get("content", "")})

        if not normalized_memories:
            response = "No memory entries found."
            reward = 1
            metrics = {}
        else:
            response_lines = []
            for mem in normalized_memories:
                title = mem.get("title") or "(untitled)"
                content = str(mem.get("content", "")).replace("\n", " ").strip()
                preview = content[:77] + "..." if len(content) > 80 else content or "(empty)"
                response_lines.append(f"- {title}: {preview}")

            response = "Memory Entries:\n" + "\n".join(response_lines)
            reward = 1  # Small positive reward for successful listing
            metrics = {}
        return ToolResponse(text=response), reward, metrics

    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class RankBM25RetrievalTool(BaseTool):
    """In-memory BM25检索工具，支持简单语料构建与相关文档返回。"""

    DEFAULT_TEXT_FIELD = "text"

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_bm25_retrieve",
                description="Search memory using keyword matching (BM25). Best for finding documents with specific terms or exact phrases. Returns JSON with ranked results.",
                parameters=OpenAIFunctionParametersSchema(
                    **{
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query used to retrieve relevant memory chunks.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of retrieved documents to return. Defaults to 20 if not specified.",
                            },
                        },
                        "required": ["query"],
                    }
                ),
            ),
        )

    async def create(self, **kwargs) -> tuple[str, ToolResponse]:
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))

        create_cfg = kwargs.get("create_kwargs") or kwargs
        chunks = create_cfg.get("chunks") if create_cfg else None
        if chunks is None:
            chunks = []
        if not isinstance(chunks, list):
            raise TypeError("'chunks' should be a list of strings.")

        documents: List[str] = []
        tokenized: List[List[str]] = []
        for chunk in chunks:
            if not isinstance(chunk, str):
                logger.warning("RankBM25RetrievalTool received non-string chunk: %s", type(chunk))
                continue
            tokens = self._tokenize(chunk)
            if tokens:
                documents.append(chunk)
                tokenized.append(tokens)

        if not hasattr(self, "_instance_dict"):
            self._instance_dict: Dict[str, Dict[str, Any]] = {}

        bm25 = BM25Okapi(tokenized) if tokenized else None
        self._instance_dict[trajectory_id] = {
            "documents": documents,
            "tokens": tokenized,
            "bm25": bm25,
        }

        if not documents:
            logger.warning("RankBM25RetrievalTool instance %s initialized with empty chunks", trajectory_id)

        return trajectory_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, arguments: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        state = getattr(self, "_instance_dict", {}).get(instance_id)
        if state is None:
            error_msg = f"Instance '{instance_id}' not initialized."
            logger.warning("[RankBM25RetrievalTool] %s", error_msg)
            payload = {"error": error_msg}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, {"error": error_msg}

        bm25 = state.get("bm25")
        documents = state.get("documents", [])
        if bm25 is None or not documents:
            payload = {"results": [], "warning": "corpus_empty"}
            metrics = {"retrieved": 0, "corpus_size": len(documents)}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, metrics

        query = arguments.get("query")
        if not query or not isinstance(query, str):
            error_msg = "Missing or invalid 'query' parameter."
            logger.warning("[RankBM25RetrievalTool] %s Received: %s", error_msg, arguments)
            payload = {"error": error_msg}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, {"error": error_msg}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            payload = {"results": [], "warning": "query_no_tokens"}
            metrics = {"retrieved": 0, "corpus_size": len(documents)}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, metrics

        top_k = min(int(arguments.get("top_k", 20)), len(documents))

        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = documents[idx]
            payload: Dict[str, Any] = {
                "rank": rank,
                "score": float(scores[idx]),
                # "doc_id": str(idx),
                "text": doc,
            }
            results.append(payload)

        response_payload = {"results": results}
        metrics = {
            "retrieved": len(results),
            "corpus_size": len(documents),
            "average_score": float(sum(item["score"] for item in results) / len(results)) if results else 0.0,
        }
        return ToolResponse(text=json.dumps(response_payload, ensure_ascii=False)), 1, metrics

    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def _tokenize(self, text: str) -> List[str]:
        translator = str.maketrans("", "", string.punctuation)
        cleaned = text.lower().translate(translator)
        return [token for token in cleaned.split() if token]


class EmbeddingRetrievalTool(BaseTool):
    """In-memory retrieval tool that uses external embedding service for semantic search."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.endpoint = os.environ['EMBEDDING_SERVICE_ENDPOINT']
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout_cfg = self.config.get("embedding_timeout", 300.0)
        timeout = aiohttp.ClientTimeout(total=float(timeout_cfg))        
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def close(self):
        await self.session.close()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="native",
            function=OpenAIFunctionSchema(
                name="memory_embedding_retrieve",
                description="Search memory using semantic similarity (embeddings). Best for finding conceptually related content when exact keywords may vary. Returns JSON with ranked results.",
                parameters=OpenAIFunctionParametersSchema(
                    **{
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query used to retrieve relevant memory documents.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of documents to return. Defaults to 20 if not specified.",
                            },
                        },
                        "required": ["query"],
                    }
                ),
            ),
        )

    async def create(self, **kwargs) -> tuple[str, ToolResponse]:
        trajectory_id = kwargs.get("create_kwargs", {}).get('trajectory_id', str(uuid4()))
        if trajectory_id in self._instance_dict:
            return trajectory_id, ToolResponse()
        chunks = kwargs.get("create_kwargs", {}).get("chunks")
        if chunks is None:
            chunks = []
        if not isinstance(chunks, list):
            raise TypeError("'chunks' should be a list of strings.")

        documents: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str) and chunk.strip():
                documents.append(chunk)
            else:
                logger.warning(
                    "EmbeddingRetrievalTool received a non-string or empty chunk: %s", type(chunk)
                )

        embeddings = None
        if documents:
            embeddings = await self._fetch_embeddings(documents)
        else:
            logger.warning("EmbeddingRetrievalTool instance %s initialized with no valid chunks", trajectory_id)

        if not hasattr(self, "_instance_dict"):
            self._instance_dict: Dict[str, Dict[str, Any]] = {}

        self._instance_dict[trajectory_id] = {
            "documents": documents,
            "embeddings": embeddings,
        }

        return trajectory_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, arguments: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        state = getattr(self, "_instance_dict", {}).get(instance_id)
        if state is None:
            error_msg = f"实例 '{instance_id}' 尚未初始化。"
            logger.warning("[EmbeddingRetrievalTool] %s", error_msg)
            payload = {"error": error_msg}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, {"error": error_msg}

        documents: List[str] = state.get("documents", [])
        embeddings = state.get("embeddings")
        if embeddings is None or not documents:
            payload = {"results": [], "warning": "corpus_empty"}
            metrics = {"retrieved": 0, "corpus_size": len(documents)}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, metrics

        query = arguments.get("query")
        if not query or not isinstance(query, str):
            error_msg = "Missing or invalid 'query' argument."
            logger.warning("[EmbeddingRetrievalTool] %s Received: %s", error_msg, arguments)
            payload = {"error": error_msg}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0, {"error": error_msg}

        query_embedding = await self._fetch_embeddings([query])
        query_embedding = query_embedding[0].unsqueeze(0)

        top_k = min(int(arguments.get("top_k", 20)), len(documents))

        hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0] if top_k else []

        results: List[Dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            corpus_idx = int(hit.get("corpus_id", -1))
            if 0 <= corpus_idx < len(documents):
                item: Dict[str, Any] = {
                    "rank": rank,
                    "text": documents[corpus_idx],
                    "score": float(hit.get("score", 0.0)),
                }
                results.append(item)

        response_payload = {"results": results}
        metrics = {
            "retrieved": len(results),
            "corpus_size": len(documents),
        }
        return ToolResponse(text=json.dumps(response_payload, ensure_ascii=False)), 1, metrics

    async def release(self, instance_id: str) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    async def _fetch_embeddings_deprecated(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("No texts provided for embedding computation.")

        model_name = self.config.get("embedding_model", self.DEFAULT_MODEL)
        payload = {"input": texts, "model": model_name}
        # timeout_cfg = self.config.get("embedding_timeout", 300.0)
        # client_timeout = aiohttp.ClientTimeout(total=float(timeout_cfg) if timeout_cfg else None)
        max_retries = 3
        backoff = 1
        for attempt in range(max_retries):
            try:
                async with self.session.post(self.endpoint, json=payload) as response:
                    if response.status != 200:
                        body = await response.text()
                        # 如果是 5xx 错误（服务端挂了），抛出异常触发重试
                        if response.status >= 500:
                            raise RuntimeError(f"Server Error {response.status}")
                        # 如果是 4xx 错误（参数不对），直接报错不重试
                        raise ValueError(f"Client Error {response.status}: {body[:200]}")
                    
                    data = await response.json()
                    break

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} retries: {exc}")
                    raise exc
                
                logger.warning(f"Retry {attempt + 1}/{max_retries} due to: {exc}. Waiting {backoff}s")
                await asyncio.sleep(backoff)
                backoff *= 2  # 指数退避，给服务端喘息机会

        vectors = data.get("data")
        if not vectors:
            raise RuntimeError("Embedding 服务返回数据为空或缺少 'data' 字段。")

        embeddings: List[torch.Tensor] = []
        for item in vectors:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if embedding is None:
                raise RuntimeError("Embedding 服务返回格式不正确，缺少 embedding 字段。")
            embeddings.append(torch.tensor(embedding, dtype=torch.float32))

        return torch.stack(embeddings)

    async def _fetch_embeddings(self, documents):
        """
        内部自动分批处理 Embedding 请求，解决 422 长度超限问题。
        """
        if not documents:
            return []

        # --- 配置区域 ---
        # 1. 单次请求的最大文本数 (Infinity 限制 2048，设 512 留余量且效率高)
        BATCH_SIZE = 512
        # 2. 并发控制：同时最多允许多少个 HTTP 请求在飞 (防止把服务端打挂)
        # 10 * 512 = 5120 条文本同时处理，这对 Infinity 来说很轻松
        CONCURRENT_LIMIT = 10 
        # ----------------
        
        sem = asyncio.Semaphore(CONCURRENT_LIMIT)
        total_docs = len(documents)
        
        # 定义处理单个 Batch 的内部函数
        async def fetch_single_batch(batch_texts):
            async with sem:  # 限制并发数
                payload = {
                    "model": self.config.get("embedding_model", self.DEFAULT_MODEL),
                    "input": batch_texts
                }
                
                # 使用你的共享 session
                # 注意：这里建议加上简单的重试逻辑，防止网络抖动
                for attempt in range(5):
                    try:
                        async with self.session.post(self.endpoint, json=payload) as response:
                            if response.status != 200:
                                # 如果是 422，说明即便分批了还是有问题（比如单条文本过长），直接抛出
                                if response.status == 422:
                                    body = await response.text()
                                    raise ValueError(f"422 Error: {body}")
                                
                                # 其他错误（500等）可以重试
                                if response.status >= 500:
                                    raise RuntimeError(f"Server Error {response.status}")
                                    
                                body = await response.text()
                                raise RuntimeError(f"Error {response.status}: {body[:200]}")
                            
                            data = await response.json()
                            # 提取 embedding 列表
                            return [item['embedding'] for item in data['data']]
                            
                    except Exception as e:
                        if attempt == 2: # 最后一次重试也失败
                            raise e
                        await asyncio.sleep(1 * (attempt + 1))

        # --- 核心逻辑：切分与 Gather ---
        
        tasks = []
        # 使用 range 函数步长切片
        for i in range(0, total_docs, BATCH_SIZE):
            # 切出当前批次
            batch = documents[i : i + BATCH_SIZE]
            # 创建任务
            tasks.append(fetch_single_batch(batch))

        # 并发执行所有批次
        # 注意：asyncio.gather 保证返回结果的顺序与 tasks 添加顺序一致
        # 所以不用担心结果乱序
        batch_results = await asyncio.gather(*tasks)
        
        # --- 结果合并 ---
        
        # batch_results 是一个二维列表：[[emb1, emb2], [emb3, emb4], ...]
        # 我们需要把它展平成一维列表：[emb1, emb2, emb3, emb4, ...]
        final_embeddings = [emb for batch in batch_results for emb in batch]
        
        # 简单的完整性校验
        if len(final_embeddings) != total_docs:
            raise RuntimeError(f"Embedding count mismatch! Sent {total_docs}, got {len(final_embeddings)}")

        return torch.tensor(final_embeddings)