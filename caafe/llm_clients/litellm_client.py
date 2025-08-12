import os
from enum import Enum
from typing import Any, Dict, List

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType

from caafe.entity.app_config import AppConfig

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class LiteLLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

        if not self.config.llm.api_key:
            raise ValueError(
                "API key not provided and LLM_API_KEY environment variable not set"
            )

        self.completion_params = {
            "model": f"{config.provider}/{config.model_name}",
            "api_key": self.config.llm.api_key,
            "base_url": self.config.llm.base_url,
            "extra_headers": self.config.llm.extra_headers,
            "max_completion_tokens": self.config.llm.max_completion_tokens,
            "temperature": self.config.llm.temperature,
            "metadata": {"session_id": self.config.session_id},
            **self.config.llm.completion_params,
        }

        if config.llm.caching.enabled:
            litellm.cache = Cache(
                type=LiteLLMCacheType.DISK, disk_cache_dir=config.caching.dir_path
            )

    def query(self, messages: str | List[Dict[str, Any]], **kwargs) -> str | None:
        """
        Query the LLM with a list of messages or a single message string.

        Args:
            messages: A list of dictionaries representing the conversation history or a single message string.

        Returns:
            The response from the LLM as a string.
        """
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        response = litellm.completion(
            messages=messages,
            **self.completion_params,
            **kwargs,
        )
        return response.choices[0].message.content
