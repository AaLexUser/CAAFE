from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from .caafe import generate_features
from .entity.task import ClassificationTask
from .metrics import auc_metric
from .preprocessing import (
    make_dataset_numeric,
    make_datasets_numeric,
    split_target_column,
)
from .run_llm_code import run_llm_code


@dataclass(frozen=True)
class Dataset:
    """Dataset metadata used by feature generation.

    Maintains positional order compatibility:
    0: name, 1: X, 2: y, 3: categorical_features, 4: columns, 5: modifications, 6: description
    """

    name: str
    X: Any
    y: Any
    categorical_features: List[int]
    columns: List[str]
    modifications: Dict[str, Any]
    description: str


class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """

    def __init__(
        self,
        base_classifier: Optional[object] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-3.5-turbo",
        n_splits: int = 10,
        n_repeats: int = 2,
        display_method: str = "markdown",
        **kwargs,
    ) -> None:
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from functools import partial

            import torch
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

            self.base_classifier = TabPFNClassifier(
                N_ensemble_configurations=16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.base_classifier.fit = partial(
                self.base_classifier.fit, overwrite_warning=True
            )
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.display_method = display_method
        self.kwargs = kwargs

    def fit_pandas(self, df, dataset_description, target_column_name, **kwargs):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        feature_columns = list(df.drop(columns=[target_column_name]).columns)

        X, y = (
            df.drop(columns=[target_column_name]).values,
            df[target_column_name].values,
        )
        return self.fit(
            X, y, dataset_description, feature_columns, target_column_name, **kwargs
        )

    def fit(self, description, dataframe, target_column):
        task = ClassificationTask(
            description=description,
            dataframe=dataframe,
            target_column=target_column,
        )

        if (
            task.dataframe.shape[0] > 3000
            and self.base_classifier.__class__.__name__ == "TabPFNClassifier"
        ):
            print(
                "WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)"
            )
        elif (
            task.dataframe.shape[0] > 10000
            and self.base_classifier.__class__.__name__ == "TabPFNClassifier"
        ):
            print("WARNING: CAAFE may take a long time to run on large datasets.")

        self.code, prompt, messages = generate_features(
            ds,
            df_train,
            model=self.llm_model,
            iterative=self.iterations,
            metric_used=auc_metric,
            iterative_method=self.base_classifier,
            display_method=self.display_method,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            **self.kwargs,  # Pass through any additional provider-specific kwarg
        )

        df_train = run_llm_code(
            self.code,
            df_train,
        )

        df_train, _, self.mappings = make_datasets_numeric(
            df_train, df_test=None, target_column=target_name, return_mappings=True
        )

        df_train, y = split_target_column(df_train, target_name)

        X, y = df_train.values, y.values.astype(int)
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.base_classifier.fit(X, y)

        # Return the classifier
        return self

    def predict_preprocess(self, X):
        """
        Helper functions for preprocessing the data before making predictions.

        Parameters:
        X (pandas.DataFrame): The DataFrame to make predictions on.

        Returns:
        numpy.ndarray: The preprocessed input data.
        """
        # check_is_fitted(self)

        if X is not pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)
        X, _ = split_target_column(X, self.target_name)

        X = run_llm_code(
            self.code,
            X,
        )

        X = make_dataset_numeric(X, mappings=self.mappings)

        X = X.values

        # Input validation
        # X = check_array(X)

        return X

    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)
