import numpy as np


class CategoricalFeatures:
    """
        A class for handling categorical features in a dataset.
    """
    def __init__(self, test_dataset, feature_index):
        """
        Initialize a CategoricalFeatures instance.

        Parameters:

        test_dataset : array-like
            The feature matrix of the dataset.

        feature_index : int
            The index of the categorical feature to be modified.
        """
        self.test_dataset = test_dataset
        self.feature_index = feature_index

    def apply(self):
        """
        Apply categorical feature transformation by replacing values with random choices.

        Returns:
            test_dataset_copy :
                array-like: A modified copy of the feature matrix with transformed categorical feature.
        """
        test_dataset_copy = np.copy(self.test_dataset)
        unique_values = [np.unique(self.test_dataset[:, self.feature_index])]
        if len(unique_values[0]) > 1:
            for i in range(self.test_dataset.shape[0]):
                old_value = self.test_dataset[i, self.feature_index]
                new_value = np.random.choice(np.setdiff1d(unique_values, [old_value]))
                test_dataset_copy[i, self.feature_index] = new_value
            return test_dataset_copy
        else:
            return test_dataset_copy
