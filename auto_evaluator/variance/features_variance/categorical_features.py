import numpy as np


class CategoricalFeatures:
    """
        A class for handling categorical features in a dataset.
    """
    def __init__(self, X_test, feature_index):
        """
        Initialize a CategoricalFeatures instance.

        Parameters:

        X_test : array-like
            The feature matrix of the dataset.

        feature_index : int
            The index of the categorical feature to be modified.
        """
        self.X_test = X_test
        self.feature_index = feature_index

    def apply(self):
        """
        Apply categorical feature transformation by replacing values with random choices.

        Returns:
            X_test_copy :
                array-like: A modified copy of the feature matrix with transformed categorical feature.
        """
        X_test_copy = np.copy(self.X_test)
        unique_values = [np.unique(self.X_test[:, self.feature_index])]
        if len(unique_values[0]) > 1:
            for i in range(self.X_test.shape[0]):
                old_value = self.X_test[i, self.feature_index]
                new_value = np.random.choice(np.setdiff1d(unique_values, [old_value]))
                X_test_copy[i, self.feature_index] = new_value
            return X_test_copy
        else:
            return X_test_copy
