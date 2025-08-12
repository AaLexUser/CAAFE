from caafe.tools.base import BaseTool
from caafe.entity.tools import Observation


class FinishTool(BaseTool):
    name: str = "finish"
    description: str = "Finish the execution of the task."
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> Observation:
        return Observation(
            is_success=True,
            message="Successfully finished the execution of the task.",
        )