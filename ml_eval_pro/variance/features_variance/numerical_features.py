import os

import numpy as np

from ml_eval_pro.configuration_manager.configuration_reader.yaml_reader import YamlReader


class NumericalFeatures:
    """
        A class for handling numerical feature manipulation based on variance analysis.
    """

    def __init__(self, X_test, feature_index, string_flag):
        """
        Initialize a NumericalFeatures instance.

        Parameters:

        - X_test : array-like or pd.DataFrame
        The feature matrix of the test dataset.

        - feature_index : int
        The index of the feature to be manipulated.

        - string_flag : bool
        A flag indicating whether the feature is of string type.
        """
        self.X_test = X_test
        self.feature_index = feature_index
        self.string_flag = string_flag
        self.__yaml_reader = YamlReader(os.path.join(os.path.curdir, "ml_eval_pro",
                                                     "config_files", "system_config.yaml"))

    def apply(self):
        """
        Apply numerical feature manipulation based on variance analysis.

        Returns:
            tuple: Two modified feature matrices:
                - The first one with added step to the feature values.
                - The second one with subtracted step from the feature values.
        """
        sorting_indices = np.argsort(self.X_test[:, self.feature_index].astype(float))
        X_test_ranked = self.X_test[sorting_indices]
        X_test_ranked_diff = np.diff(X_test_ranked[:, self.feature_index].astype(float))
        std_dev = np.std(X_test_ranked_diff)
        step = std_dev * self.__yaml_reader.get('variance')['step_size']
        return self.__apply_step(X_test_ranked, step), self.__apply_step(X_test_ranked, step, "-")

    def __apply_step(self, X_test_ranked, step, sign: str = '+'):
        """
        Apply a specified step to the ranked feature values.

        Parameters:
        X_test_ranked : array-like
            The feature matrix sorted based on the specified feature.

        step : float
            The step to be applied to the feature values.

        sign : str, optional (default='+')
            The sign of the step. If '+', step will be added; if '-', step will be subtracted.

        Returns:
            array-like: The modified feature matrix with the applied step.
        """
        global feature_with_step
        if self.string_flag:
            X_test_ranked_copy = np.copy(X_test_ranked)
        else:
            X_test_ranked_copy = np.copy(X_test_ranked).astype(float)
        if sign == '+':
            feature_with_step = X_test_ranked_copy[:, self.feature_index].astype(float) + step
        elif sign == '-':
            feature_with_step = X_test_ranked_copy[:, self.feature_index].astype(float) - step
        X_test_ranked_copy[:, self.feature_index] = feature_with_step
        return X_test_ranked_copy
