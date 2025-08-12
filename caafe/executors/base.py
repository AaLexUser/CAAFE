from typing import Protocol

from caafe.entity.tools import Observation



class Executor(Protocol):

    def run(self, code: str) -> Observation:
        """Run the provided code."""
        ...