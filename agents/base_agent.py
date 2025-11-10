from typing import List
from openai import OpenAI, BadRequestError
import uuid

from dotenv import load_dotenv
load_dotenv(".env") # 默认会找项目根目录的 .env
import os


MODEL_NAME_MAP = {
    "gpt4.1": "azure-gpt-4_1",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
}

class BaseAgent:
    def __init__(
        self,
        client: OpenAI = OpenAI(
            base_url="http://api-hub.inner.chj.cloud/llm-gateway/v1",
            api_key="sk-", # 使用一个填充值，实际上我们用headers传递认证信息
            default_headers={
                "BCS-APIHub-RequestId": str(uuid.uuid4()),  # 将在请求时动态生成
                "X-CHJ-GWToken": os.getenv("X-CHJ-GWToken"),
                "X-CHJ-GW-SOURCE": os.getenv("X-CHJ-GW-SOURCE"),
            }
        ),
        model_name: str = "gpt4.1"
    ):
        self.client = client
        self.model_name = model_name
    
    def _handle_api_error(self, error: Exception, query: str = "") -> str:
        """Handle API errors and return appropriate error messages"""
        if isinstance(error, BadRequestError):
            error_details = str(error)
            if "context_length_exceeded" in error_details or "maximum context length" in error_details:
                return f"ERROR_CONTEXT_LENGTH_EXCEEDED: The input is too long for the model to process. Query: {query[:100]}..."
            elif "content_filter" in error_details or "ResponsibleAIPolicyViolation" in error_details:
                return f"ERROR_CONTENT_FILTER: The content was filtered by Azure OpenAI's policy. Query: {query[:100]}..."
            else:
                return f"ERROR_BAD_REQUEST: {error_details}"
        else:
            return f"ERROR_API_CALL: {str(error)}"
    
    def add_memory(
        self,
        chunk: str
    ):
        raise NotImplementedError

    def QA(
        self,
        query: str
    ) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state between samples."""
        # Default implementation does nothing; subclasses may override.
        return None

    def prepare_sample(self, sample) -> None:
        """Optional hook to configure the agent before processing a sample."""
        # Subclasses can override when sample-level context is required.
        return None