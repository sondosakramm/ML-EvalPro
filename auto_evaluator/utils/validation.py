import numpy as np
import pandas as pd


def check_consistent_length(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")


def check_numpy(data):
    try:
        if not isinstance(data, np.ndarray):
            return data.to_numpy()
        return data
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e