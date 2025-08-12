from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .entity.task import ClassificationTask
from .run_llm_code import run_llm_code
from .caafe_evaluate import evaluate_dataset


@dataclass
class ToolResult:
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
        except Exception as e:  # noqa: BLE001
            return ToolResult(name=name, ok=False, content=str(e))


registry = ToolRegistry()


# --------------------- Tools --------------------- #

def load_dataset(path: str = "titanic/train.csv", target: str = "Survived", limit: int = 1000) -> ToolResult:
    df = pd.read_csv(path)
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=0)
    if target not in df.columns:
        return ToolResult("load_dataset", ok=False, content=f"Target column '{target}' not in dataframe")
    description = f"Loaded dataset from {path} with shape {df.shape}. Target column '{target}'."  # placeholder; could be enriched
    task = ClassificationTask(description=description, train_df=df, target_column=target)
    preview = df.head(5).to_dict(orient="list")
    return ToolResult(
        name="load_dataset",
        ok=True,
        content=f"Dataset loaded. Columns: {list(df.columns)}",
        data={"task": task, "preview": preview},
    )


def generate_features_and_score(code: str, task: ClassificationTask, test_size: float = 0.2, method: str = "logistic", metric_used: str = "acc") -> ToolResult:
    # Split
    df = task.train_df.copy()
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=0, stratify=df[task.target_column])

    # Baseline evaluation
    baseline_res = evaluate_dataset(
        df_train=train_df.copy(),
        df_test=test_df.copy(),
        prompt_id="baseline",
        name="dataset",
        method=method,
        metric_used=metric_used,
        target_name=task.target_column,
    )

    # Apply feature code
    try:
        transformed_train = run_llm_code(code, train_df.drop(columns=[task.target_column]))
        transformed_test = run_llm_code(code, test_df.drop(columns=[task.target_column]))
    except Exception as e:  # noqa: BLE001
        return ToolResult(
            name="generate_features_and_score",
            ok=False,
            content=f"Feature code failed: {type(e).__name__} {e}",
            data={"error_type": type(e).__name__},
        )

    # Reattach target
    transformed_train[task.target_column] = train_df[task.target_column].values
    transformed_test[task.target_column] = test_df[task.target_column].values

    feat_res = evaluate_dataset(
        df_train=transformed_train,
        df_test=transformed_test,
        prompt_id="features",
        name="dataset",
        method=method,
        metric_used=metric_used,
        target_name=task.target_column,
    )

    improvement_acc = feat_res["acc"] - baseline_res["acc"]
    improvement_roc = feat_res["roc"] - baseline_res["roc"]

    return ToolResult(
        name="generate_features_and_score",
        ok=True,
        content=f"Baseline ACC {baseline_res['acc']:.3f} ROC {baseline_res['roc']:.3f}; With features ACC {feat_res['acc']:.3f} ROC {feat_res['roc']:.3f}; ΔACC {improvement_acc:.3f} ΔROC {improvement_roc:.3f}",
        data={
            "baseline": baseline_res,
            "with_features": feat_res,
            "improvement_acc": improvement_acc,
            "improvement_roc": improvement_roc,
        },
    )


registry.register("load_dataset", load_dataset)
registry.register("generate_features_and_score", generate_features_and_score)

__all__ = ["registry", "load_dataset", "generate_features_and_score", "ToolResult", "ToolRegistry"]
