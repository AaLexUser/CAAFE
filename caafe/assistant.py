from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .entity.task import ClassificationTask
from .entity.app_config import AppConfig
from .llm_clients.litellm_client import LiteLLMClient
from .prompting.utils import parse_tool_calls, extract_code
from .tools import registry, ToolResult
from .executors.jupyter import JupyterExecutor
from .tools import jupyter as jupyter_module
from .tools import finish as finish_module
from .prompting.sections.tool_available import tool_use_section


def _render_tools_block() -> str:
    lines = [tool_use_section().rstrip(), ""]
    # Reflect over registered tools (internal attribute access acceptable inside package)
    for name, func in getattr(registry, "_tools", {}).items():  # type: ignore[attr-defined]
        doc = (func.__doc__ or "").strip()
        try:
            import inspect

            sig = str(inspect.signature(func))
        except Exception:  # noqa: BLE001
            sig = "(…)"
        lines.append(f"### {name}\nSignature: {name}{sig}\nDescription: {doc if doc else 'No description.'}\nUsage XML example:\n<{name}><param>value</param></{name}>\n")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a ReAct agent. You think step-by-step. To use a tool, output an XML block like: "
    "<tool_name><param>value</param></tool_name>. Provide reasoning before tool call."
)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class AssistantState:
    config: AppConfig
    task: Optional[ClassificationTask] = None
    history: List[Message] = field(default_factory=list)
    scratchpad: List[str] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.history.append(Message(role=role, content=content))

    def messages_for_llm(self) -> List[Dict[str, str]]:
        # Always regenerate tools block so new registrations appear.
        tools_block = _render_tools_block()
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tools_block},
        ]
        for m in self.history:
            msgs.append({"role": m.role, "content": m.content})
        return msgs


class AssistantAgent:
    """Hexagonal core orchestrating LLM (port) and tools (port)."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.state = AssistantState(config=config or AppConfig())
        self.llm = LiteLLMClient(self.state.config)
        # Initialize async-capable tools (Jupyter / Finish) for potential future unified interface
        self.jupyter_tool = jupyter_module.JupyterTool(executor=JupyterExecutor(workspace=__import__("pathlib").Path(".")))
        self.finish_tool = finish_module.FinishTool()

    # ---------------- Public API ---------------- #
    def run(self, user_goal: str, max_steps: int = 8) -> AssistantState:
        self.state.add("user", user_goal)
        for step in range(max_steps):
            llm_output = self.llm.query(self.state.messages_for_llm()) or ""
            self.state.add("assistant", llm_output)
            tool_calls = parse_tool_calls(llm_output)
            if not tool_calls:
                # No tool call => assume final answer
                break
            for call in tool_calls:
                tool_name = call["name"]
                args = call.get("args", {})
                # Inject task if needed
                if self.state.task and "task" not in args and tool_name != "load_dataset":
                    args["task"] = self.state.task
                # Support synchronous registry tools first
                result: ToolResult
                if tool_name == self.jupyter_tool.name:
                    # For now execute synchronously via loop (could refactor to async run())
                    import asyncio

                    async def _run():  # noqa: D401
                        obs = await self.jupyter_tool.execute(code=args.get("code", ""), language=args.get("language", "python"))  # type: ignore[arg-type]
                        return ToolResult(
                            name=self.jupyter_tool.name,
                            ok=obs.is_success,
                            content=obs.message,
                            data={"images": obs.base64_images},
                        )

                    result = asyncio.run(_run())
                elif tool_name == self.finish_tool.name:
                    import asyncio

                    async def _run_finish():  # noqa: D401
                        obs = await self.finish_tool.execute()  # type: ignore[call-arg]
                        return ToolResult(
                            name=self.finish_tool.name,
                            ok=obs.is_success,
                            content=obs.message,
                        )

                    result = asyncio.run(_run_finish())
                else:
                    result = registry.run(tool_name, **args)
                if result.name == "load_dataset" and result.ok:
                    self.state.task = result.data.get("task")  # type: ignore[arg-type]
                observation = result.to_observation_message()
                self.state.add("user", f"Observation: {observation}\nContinue reasoning. If goal achieved, provide final answer.")
                if tool_name == "generate_features_and_score" and result.ok:
                    # supply code feedback suggestion prompt
                    pass
        return self.state

    # Convenience for feature generation directly
    def generate_features(self, code: str) -> ToolResult:
        if not self.state.task:
            raise ValueError("No task loaded. Use load_dataset first.")
        return registry.run("generate_features_and_score", code=code, task=self.state.task)


__all__ = ["AssistantAgent", "AssistantState", "Message"]
