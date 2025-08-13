import copy

import pandas as pd
from sklearn.base import BaseEstimator

from caafe.metrics import accuracy_metric as tabpfn_accuracy_metric
from caafe.metrics import auc_metric as tabpfn_auc_metric

from .data import get_X_y
from .preprocessing import make_dataset_numeric, make_datasets_numeric


def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    prompt_id,
    name,
    method,
    metric_used,
    target_name,
    max_time=300,
    seed=0,
):
    """
    Minimal evaluator used during iterative feature generation. Supports only
    scikit-learn compatible estimators (method: BaseEstimator).
    """
    df_train, df_test = copy.deepcopy(df_train), copy.deepcopy(df_test)
    df_train, _, mappings = make_datasets_numeric(
        df_train, None, target_name, return_mappings=True
    )
    df_test = make_dataset_numeric(df_test, mappings=mappings)

    if df_test is not None:
        test_x, test_y = get_X_y(df_test, target_name=target_name)

    x, y = get_X_y(df_train, target_name=target_name)

    # Fit and predict using provided estimator
    if isinstance(method, BaseEstimator):
        method.fit(X=x.numpy(), y=y.numpy().astype(int))
        ys = method.predict_proba(test_x.numpy())
    else:
        raise ValueError(
            "Only scikit-learn compatible estimators are supported in this simplified evaluator."
        )

    acc = tabpfn_accuracy_metric(test_y, ys)
    roc = tabpfn_auc_metric(test_y, ys)

    method_str = method if isinstance(method, str) else "transformer"
    return {
        "acc": float(acc.numpy()),
        "roc": float(roc.numpy()),
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": x.shape[-1],
    }
