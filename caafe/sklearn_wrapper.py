from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from .caafe import generate_features
from .metrics import auc_metric
from .preprocessing import (
    make_dataset_numeric,
    make_datasets_numeric,
    split_target_column,
)
from .run_llm_code import run_llm_code


class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    """
    CAAFEClassifier(base_classifier=None, optimization_metric='accuracy', iterations=10, n_splits=10, n_repeats=2, display_method='markdown', config_overrides=None, config_path=None, presets=None, **kwargs)
    A scikit-learn compatible classifier that uses the CAAFE algorithm to iteratively synthesize and evaluate features with an LLM, then trains a downstream (base) classifier on the transformed dataset.
    The workflow is:
    1) Optionally generate feature-engineering code with an LLM (CAAFE).
    2) Apply the generated code to transform the training data.
    3) Numerically encode the transformed data in a stable way for train/inference.
    4) Fit a base classifier on the engineered features.
    5) At prediction time, re-apply the exact same transformation and encoding before calling the base classifier.
    Parameters
    ----------
    base_classifier : object, optional
        A scikit-learn compatible estimator implementing fit, predict, and optionally predict_proba.
        If None, a TabPFNClassifier is created with a device chosen automatically (CUDA if available).
    optimization_metric : {'accuracy', 'auc'}, default='accuracy'
        Metric name intended to guide the feature generation/selection process.
        Note: The current implementation may use an internal metric configuration during iteration.
    iterations : int, default=10
        Number of CAAFE iterations (feature-generation/selection rounds).
    n_splits : int, default=10
        Number of cross-validation splits used to evaluate candidate features during generation.
    n_repeats : int, default=2
        Number of repeated cross-validation cycles used in evaluation.
    display_method : str, default='markdown'
        Display mode for any intermediate outputs/logs produced during feature generation.
    config_overrides : list of str, optional
        Provider-/framework-specific override strings passed to the feature generation backend.
    config_path : str, optional
        Path to configuration files for the feature generation backend.
    presets : str, optional
        Named preset to control the behavior of the feature generation backend.
    **kwargs
        Additional provider-specific keyword arguments forwarded to the feature generation routine
        (for example, model name, temperature, timeouts, etc.).
    Attributes
    ----------
    base_classifier : object
        The wrapped estimator trained on engineered features.
    iterations : int
        Number of CAAFE iterations used.
    optimization_metric : str
        Optimization metric name provided at initialization.
    n_splits : int
        Cross-validation splits used during feature generation.
    n_repeats : int
        Number of repeated CV cycles used during feature generation.
    display_method : str
        Display mode used by the feature generation backend.
    config_overrides : list of str or None
        Overrides passed to the generation backend.
    config_path : str or None
        Path to generation backend configs.
    presets : str or None
        Named preset used by the generation backend.
    kwargs : dict
        Extra arguments forwarded to the generation backend.
        Description of the dataset as passed to fit or fit_pandas.
    feature_names : list of str
        Original feature names provided at fit time.
        Name of the target variable.
    X_ : array-like of shape (n_samples, n_features)
        Original training features provided to fit.
    y_ : array-like of shape (n_samples,)
        Original training targets provided to fit.
    code : str
        The generated Python code (if any) that performs feature engineering. Empty when disable_caafe=True.
    mappings : dict
        Encoding mappings learned during training that are reused at inference to ensure consistent numeric representations.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during fit (as per scikit-learn convention).
    Methods
    -------
    fit_pandas(df, dataset_description, target_column_name, **kwargs)
        Convenience method to fit from a pandas DataFrame, inferring feature names and target split.
    fit(X, y, dataset_description, feature_names, target_name, disable_caafe=False)
        Fit the classifier. When disable_caafe=True, no LLM-generated features are created and the base classifier
        is trained on the original features after numeric encoding.
    predict(X)
        Predict class labels for X after applying the same feature-generation code (if any) and encodings.
    predict_proba(X)
        Predict class probabilities for X (if supported by the base classifier).
    predict_preprocess(X)
        Internal helper that applies the stored transformation code and encodings to raw inputs prior to prediction.
    Notes
    -----
    - For large datasets, the default TabPFN backend can be slow. Consider providing a different base_classifier such as RandomForestClassifier.
    - To avoid column-order mismatches at prediction time, prefer passing a pandas DataFrame with the same column names used during training.
    - If a target column is accidentally present during prediction, it will be ignored/removed by the preprocessing step.
    Examples
    --------
    Basic usage with a pandas DataFrame:
    >>> clf = CAAFEClassifier(iterations=5, n_splits=5, n_repeats=1)
    >>> clf.fit_pandas(df, dataset_description="Binary classification on tabular data", target_column_name="label")
    >>> y_pred = clf.predict(df.drop(columns=["label"]))
    >>> y_proba = clf.predict_proba(df.drop(columns=["label"]))
    Using arrays and explicit feature/target names:
    >>> X, y = X_train.values, y_train.values
    >>> feature_names = list(X_train.columns)
    >>> clf = CAAFEClassifier()
    >>> clf.fit(X, y, dataset_description="Demo dataset", feature_names=feature_names, target_name="label")
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        base_classifier: Optional[object] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        n_splits: int = 10,
        n_repeats: int = 2,
        display_method: str = "markdown",
        config_overrides: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        presets: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from functools import partial

            import torch
            from tabpfn import TabPFNClassifier

            self.base_classifier = TabPFNClassifier(
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.base_classifier.fit = partial(
                self.base_classifier.fit, overwrite_warning=True
            )
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.display_method = display_method
        self.config_overrides = config_overrides
        self.config_path = config_path
        self.presets = presets
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

    def fit(self, X, y, dataset_description, feature_names, target_name):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name

        self.X_ = X
        self.y_ = y

        if (
            X.shape[0] > 3000
            and self.base_classifier.__class__.__name__ == "TabPFNClassifier"
        ):
            print(
                "WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)"
            )
        elif (
            X.shape[0] > 10000
            and self.base_classifier.__class__.__name__ == "TabPFNClassifier"
        ):
            print("WARNING: CAAFE may take a long time to run on large datasets.")

        ds = [
            "dataset",
            X,
            y,
            [],
            self.feature_names + [target_name],
            {},
            dataset_description,
        ]
        # Add X and y as one dataframe
        df_train = pd.DataFrame(
            X,
            columns=self.feature_names,
        )
        df_train[target_name] = y

        self.code, prompt, messages = generate_features(
            ds,
            df_train,
            iterative=self.iterations,
            metric_used=auc_metric,
            iterative_method=self.base_classifier,
            display_method=self.display_method,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            config_overrides=self.config_overrides,
            config_path=self.config_path,
            presets=self.presets,
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
