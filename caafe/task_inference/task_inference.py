from typing import Protocol

from ..entity.task import ClassificationTask
from ..llm_clients.base_client import LLMClient
from ..prompting.prompt_generator import FeatureGenerationPromptGenerator


class TaskInference(Protocol):
    def transform(self, task: ClassificationTask) -> ClassificationTask:
        # Implement the transformation logic here
        return task


class LoadTaskInference:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def transform(self, task: ClassificationTask) -> ClassificationTask:  # noqa: D401
        """Load dataset files (already provided in task). Placeholder for enrichment."""
        # Could query LLM to refine description; keep simple for now.
        return task


class FeatureGeneration:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def transform(self, task: ClassificationTask) -> ClassificationTask:
        """Generate new features using LLM model and update task DataFrame."""
        pg = FeatureGenerationPromptGenerator(
            description=task.description,
            train_df=task.train_df,
            target_column=task.target_column,
            how_many=1,
        )
        messages = pg.generate_chat_prompt()
        response = self.llm.query(messages)
        if not response:
            return task
        code = pg.parser(response)
        try:
            from ..run_llm_code import run_llm_code

            new_df = run_llm_code(code, task.train_df.drop(columns=[task.target_column]))
            new_df[task.target_column] = task.train_df[task.target_column].values
            task.train_df = new_df
        except Exception:
            # Leave task unchanged on failure
            pass
        return task
