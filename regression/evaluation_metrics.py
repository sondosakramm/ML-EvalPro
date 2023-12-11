import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - mae: float
        Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - mape: float
        Mean Absolute Percentage Error.
    """
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(absolute_percentage_errors) * 100
    return mape


def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - mse: float
        Mean Squared Error.
    """
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - rmse: float
        Root Mean Squared Error.
    """
    curr_mse = mse(y_true, y_pred)
    return np.sqrt(curr_mse)


def medae(y_true, y_pred):
    """
    Calculate Median Absolute Error.

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - medae: float
        Median Absolute Error.
    """
    return median_absolute_error(y_true, y_pred)


def mean_bias_deviation(y_true, y_pred):
    """
    Calculate Mean Bias Deviation.

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - mean_bias_deviation: float
        Mean Bias Deviation.
    """
    return np.mean(y_pred - y_true)


def r_square_error(y_true, y_pred):
    """
    Calculate R-Squared (Coefficient of Determination).

    Parameters:
    - y_true: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like, shape (n_samples,)
        Estimated target values.

    Returns:
    - r_squared: float
        R-Squared value.
    """
    return r2_score(y_true, y_pred)
