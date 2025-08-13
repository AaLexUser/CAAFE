from typing import Any, Protocol


class LLMClient(Protocol):
    def query(self, messages: str | list[dict[str, Any]], **kwargs) -> str | None:
        """Query the LLM with a list of messages or a single message string."""
        ...
