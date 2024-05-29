import numpy as np
import pandas as pd

from ml_eval_pro.utils.validation import convert_dataframe_to_numpy
from ml_eval_pro.variance.features_variance.categorical_features import CategoricalFeatures
from ml_eval_pro.variance.features_variance.numerical_features import NumericalFeatures
from ml_eval_pro.variance.model_var.model_variance import ModelVariance


class ModelVarianceByTestData(ModelVariance):
    """
    A class for measuring the model variance by using test data.
    """

    def __init__(self, model, test_dataset, target, model_type: str, evaluation_metric: str, step_size: float, threshold: float):
        """
        Initialize a ModelVarianceByTrainTestData instance.

        Parameters:

        model : object, The machine learning model to assess for variance.

        test_dataset : array-like or pd.DataFrame The feature matrix of the test dataset.

        target : array-like or pd.Series The true labels or target values for the test dataset.

        train_dataset : array-like or pd.DataFrame, The feature matrix of the training dataset.

        train_target : array-like or pd.Series, The true labels or target values for the training dataset.

        model_type (str): str, optional (default='regression') The type of problem,
            either 'regression' or 'classification'.

        evaluation_metric (str): str, optional (default='MAE') The evaluation metric to be used for calculating errors.
         Defaults to 'MAE' (Mean Absolute Error) for regression problems.

         step_size (float): The step size to be used when adjusting variance.

         threshold (float): The variance threshold for assessing model variance.
        """
        self.step_size = step_size
        self.threshold = threshold
        self.model_avg_error = []
        self.test_dataset_with_features_name = test_dataset
        super().__init__(model=model, model_type=model_type, test_dataset=convert_dataframe_to_numpy(test_dataset),
                         target=convert_dataframe_to_numpy(target), evaluation_metric=evaluation_metric)


    def predict_with_column_names(self, data):
        """
        Predict using the model with column names.

        Parameters:
        data : np.ndarray
            The feature matrix for prediction.

        Returns:
        np.ndarray or pd.DataFrame:
            The predictions with column names.
        """
        if isinstance(data, np.ndarray):
            data_df = pd.DataFrame(data, columns=self.test_dataset_with_features_name.columns)
            return self.model.predict(data_df)
        elif isinstance(data, pd.DataFrame):
            return self.model.predict(data)
        else:
            raise ValueError("Data must be either a numpy array or a DataFrame.")


    def __get_numerical(self, feature_index, string_flag):
        """Get test_dataset after doing small perturbations on numerical features"""
        num = NumericalFeatures(self.test_dataset, feature_index, string_flag, self.step_size)
        step_added_arr, step_subtracted_arr = num.apply()
        avg_predictions = np.mean([self.calculate_errors(self.target,
                                                         self.predict_with_column_names(step_added_arr)),
                                   self.calculate_errors(self.target,
                                                         self.predict_with_column_names(step_subtracted_arr))], axis=0)
        self.model_avg_error.append(avg_predictions)

    def __get_categorical(self, feature_index):
        """Get test_dataset after doing small perturbations on categorical features"""
        cat = CategoricalFeatures(self.test_dataset, feature_index)
        cat_arr = cat.apply()
        self.model_avg_error.append(self.calculate_errors(self.target,
                                                          self.predict_with_column_names(cat_arr)))

    def calculate_variance(self):
        """
        a method that calculate the variance.
        """
        self.model_avg_error.append(self.calculate_errors(self.target,
                                                          self.predict_with_column_names(self.test_dataset)))
        for feature_index in range(self.test_dataset.shape[1]):
            feature = self.test_dataset[:, feature_index]
            if np.issubdtype(feature.dtype, np.integer) or np.issubdtype(feature.dtype, np.floating):
                self.__get_numerical(feature_index, False)
            else:
                if np.char.isnumeric(feature).all() or np.char.isdecimal(feature).all():
                    self.__get_numerical(feature_index, True)
                else:
                    self.__get_categorical(feature_index)

    def get_diff(self):
        """
        Get the difference between test_dataset predictions error and each feature prediction error after perturbations.

        return:
         list: difference between test_dataset predictions error and each feature prediction error that exceeds the
               threshold,
               with the names of the features instead of indexes.
        """
        exceeds_threshold = []
        feature_names = self.test_dataset_with_features_name.columns.tolist()  # Convert column names to a list
        for i in range(len(self.model_avg_error[1:])):
            if abs(self.model_avg_error[i] - self.model_avg_error[0]) > self.threshold:
                exceeds_threshold.append(feature_names[i])
        return exceeds_threshold
