def check_consistent_length(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")