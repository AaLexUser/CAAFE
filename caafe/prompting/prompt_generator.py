from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from .utils import extract_code


class PromptGenerator(ABC):
    def __init__(self):
        self.parser = self.create_parser()

    @property
    def system_prompt(self):
        return "You are an expert assistant that parses information about data science tasks, such as data science competitions."

    @abstractmethod
    def generate_prompt(self) -> str:
        """Generate a prompt string."""
        ...

    def generate_chat_prompt(self):
        chat_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.generate_prompt()},
        ]

        return chat_prompt

    def create_parser(self):
        return extract_code


class FeatureGenerationPromptGenerator(PromptGenerator):
    def __init__(
        self,
        description: str,
        train_df: pd.DataFrame,
        target_column: str,
        how_many: int = 1,
    ):
        super().__init__()
        self.description = description
        self.train_df = train_df
        self.target_column = target_column
        self.how_many = how_many

    @property
    def system_prompt(self) -> str:
        return "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible."

    def generate_prompt(self) -> str:
        df_head = self.train_df.head(10)

        nan_freq_pct = self.train_df.isna().mean() * 100

        lines = []
        for col in df_head.columns:
            samples = df_head[col].tolist()
            if is_float_dtype(self.train_df[col].dtype):
                samples = [
                    round(x, 2) if isinstance(x, (int, float, np.floating)) else x
                    for x in samples
                ]
            lines.append(
                f"{col} ({self.train_df[col].dtype}): NaN-freq [{nan_freq_pct[col]:.2g}%], Samples {samples}"
            )

        samples_str = "\n".join(lines)

        how_many = (
            f"up to {self.how_many} useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
            if self.how_many == 1
            else "exactly one useful column"
        )

        return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{self.description}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples_str}
    
This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(self.train_df))}

This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{self.target_column}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{self.target_column}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{self.train_df.columns[0]}': {list(self.train_df.iloc[:3, 0].values)}, '{self.train_df.columns[1]}': {list(self.train_df.iloc[:3, 1].values)}, ...)
(Some pandas code using {self.train_df.columns[0]}', '{self.train_df.columns[1]}', ... to add a new column for each row in df)
```

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
"""
