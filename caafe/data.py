import torch


def get_X_y(df_train, target_name):
    """Return (x, y) torch tensors from a pandas DataFrame and target column name.

    Only functionality required by the CAAFE notebook use-case.
    """
    y = torch.tensor(df_train[target_name].astype(int).to_numpy())
    x = torch.tensor(df_train.drop(target_name, axis=1).to_numpy())
    return x, y
