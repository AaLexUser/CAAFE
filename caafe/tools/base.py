from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from caafe.entity.tools import Observation
import json


class ToolResult(BaseModel):
    name: str
    ok: bool
    content: str
    data: Dict[str, Any] | None = None

    def to_observation_message(self) -> str:
        status = "SUCCESS" if self.ok else "ERROR"
        payload = {k: v for k, v in (self.data or {}).items() if k not in {"df", "task"}}
        return f"Tool {self.name} => {status}: {self.content}\nMeta: {json.dumps(payload, ensure_ascii=False)}"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}

    def register(self, name: str, func):
        self._tools[name] = func

    def has(self, name: str) -> bool:
        return name in self._tools

    def run(self, name: str, **kwargs) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name=name, ok=False, content=f"Unknown tool '{name}'")
        try:
            return self._tools[name](**kwargs)
        except Exception as e:
            return ToolResult(name=name, ok=False, content=str(e))

class BaseTool(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: Optional[dict] = None


    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Observation:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }