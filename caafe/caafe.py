import copy
from typing import List, Optional

import numpy as np
from sklearn.model_selection import RepeatedKFold

from caafe.configs.loader import load_config
from caafe.llm_clients.litellm_client import LiteLLMClient

from .caafe_evaluate import evaluate_dataset
from .prompting.prompt_generator import FeatureGenerationPromptGenerator
from .run_llm_code import run_llm_code


def generate_features(
    ds,
    df,
    iterative: int = 1,
    metric_used: Optional[str] = None,
    iterative_method: str = "logistic",
    display_method: str = "markdown",
    n_splits: int = 10,
    n_repeats: int = 2,
    config_overrides: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    presets: Optional[str] = None,
    **kwargs,
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        try:
            import importlib

            ipy = importlib.import_module("IPython.display")

            def display_markdown(x):
                return ipy.display(ipy.Markdown(x))

            display_method = display_markdown
        except Exception:
            display_method = print
    else:
        display_method = print

    assert iterative == 1 or metric_used is not None, (
        "metric_used must be set if iterative"
    )

    config = load_config(
        presets=presets, overrides=config_overrides, config_path=config_path
    )

    llm_client = LiteLLMClient(config)

    def execute_and_evaluate_code_block(full_code, code):
        old_accs, old_rocs, accs, rocs = [], [], [], []

        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        for train_idx, valid_idx in ss.split(df):
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            # Remove target column from df_train
            target_train = df_train[ds[4][-1]]
            target_valid = df_valid[ds[4][-1]]
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)

            try:
                df_train = run_llm_code(
                    full_code,
                    df_train,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_valid = run_llm_code(
                    full_code,
                    df_valid,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_train_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_train_extended,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_valid_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_valid_extended,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )

            except Exception as e:
                display_method(f"Error in code execution. {type(e)} {e}")
                display_method(f"```python\n{format_for_display(code)}\n```\n")
                return e, None, None, None, None

            # Add target column back to df_train
            df_train[ds[4][-1]] = target_train
            df_valid[ds[4][-1]] = target_valid
            df_train_extended[ds[4][-1]] = target_train
            df_valid_extended[ds[4][-1]] = target_valid

            import os
            import sys

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result_old = evaluate_dataset(
                        df_train=df_train,
                        df_test=df_valid,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )

                    result_extended = evaluate_dataset(
                        df_train=df_train_extended,
                        df_test=df_valid_extended,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )
                finally:
                    sys.stdout = old_stdout

            old_accs += [result_old["roc"]]
            old_rocs += [result_old["acc"]]
            accs += [result_extended["roc"]]
            rocs += [result_extended["acc"]]
        return None, rocs, accs, old_rocs, old_accs

    prompt_generator = FeatureGenerationPromptGenerator(
        description=ds[-1], train_df=df, target_column=ds[4][-1], how_many=iterative
    )
    display_method(f"*Dataset description:*\n {ds[-1]}")

    n_iter = iterative
    full_code = ""
    messages = prompt_generator.generate_chat_prompt()
    # Keep a copy of the textual prompt for returning/saving
    prompt_text = prompt_generator.generate_prompt()
    i = 0
    while i < n_iter:
        try:
            response = llm_client.query(messages, **kwargs)
            code = prompt_generator.parser(response)
            if not code:
                display_method("No code generated. Stopping.")
                break
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        e, rocs, accs, old_rocs, old_accs = execute_and_evaluate_code_block(
            full_code, code
        )
        if e is not None:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                ```python
                                """,
                },
            ]
            continue

        improvement_roc = np.nanmean(rocs) - np.nanmean(old_rocs)
        improvement_acc = np.nanmean(accs) - np.nanmean(old_accs)

        add_feature = True
        add_feature_sentence = "The code was executed and changes to ´df´ were kept."
        if improvement_roc + improvement_acc <= 0:
            add_feature = False
            add_feature_sentence = f"The last code changes to ´df´ were discarded. (Improvement: {improvement_roc + improvement_acc})"

        display_method(
            "\n"
            + f"*Iteration {i}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n"
            + f"Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n"
            + f"Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n"
            + f"{add_feature_sentence}\n"
            + "\n"
        )

        if len(code) > 10:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}
Next codeblock:
""",
                },
            ]
        if add_feature:
            full_code += code

    return full_code, prompt_text, messages
