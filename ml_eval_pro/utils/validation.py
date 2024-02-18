import numpy as np
import pandas as pd


def check_consistent_length(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")


def convert_dataframe_to_numpy(data):
    try:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        if isinstance(data, pd.Series):
            return data.to_numpy()
        if isinstance(data, np.ndarray):
            return data
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")