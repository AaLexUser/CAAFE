import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class ClassificationTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    description: str
    train_df: pd.DataFrame
    target_column: str

    @model_validator(mode="after")
    def _validate_target_column_present(self):
        if self.target_column not in self.train_df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in the train DataFrame."
            )
        return self

    @property
    def target(self) -> pd.Series:
        return self.train_df[self.target_column]
