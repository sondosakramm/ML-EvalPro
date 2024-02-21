import numpy as np

from ml_eval_pro.utils.validation import convert_dataframe_to_numpy
from ml_eval_pro.variance.features_variance.categorical_features import CategoricalFeatures
from ml_eval_pro.variance.features_variance.numerical_features import NumericalFeatures
from ml_eval_pro.variance.model_var.model_variance import ModelVariance


class ModelVarianceByTestData(ModelVariance):
    """
    A class for measuring the model variance by using test data only.

    Attributes:
        model_avg_error:
            list: that saves the calculated errors.
        X_test_with_features_name:
             array-like or pd.DataFrame: to save the original X_test.
    """

    def __init__(self, model, X_test, y_test, problem_type='regression', metric='MAE'):
        """
        Initialize a ModelVarianceByTestData instance.

        Parameters:

        model: model that it's variance will be calculated.

        X_test: array-like or pd.DataFrame
            The feature matrix of the test dataset.

        y_test: array-like or pd.Series
            The true labels or target values for the test dataset.

        problem_type : str, optional (default='regression') The type of problem,
            either 'regression' or 'classification'.

        metric : str
            optional (default='MAE') The evaluation metric to be used for calculating errors.
            Defaults to 'MAE' (Mean Absolute Error) for regression problems.
        """
        self.model_avg_error = []
        self.X_test_with_features_name = X_test
        super().__init__(model, convert_dataframe_to_numpy(X_test), convert_dataframe_to_numpy(y_test), None, None,
                         problem_type, metric)

    def __get_numerical(self, feature_index, string_flag):
        """Get X_test after doing small perturbations on numerical features"""
        num = NumericalFeatures(self.X_test, feature_index, string_flag)
        step_added_arr, step_subtracted_arr = num.apply()
        avg_predictions = np.mean([self.calculate_errors(self.y_test, self.model.predict(step_added_arr)),
                                   self.calculate_errors(self.y_test, self.model.predict(step_subtracted_arr))], axis=0)
        self.model_avg_error.append(avg_predictions)

    def __get_categorical(self, feature_index):
        """Get X_test after doing small perturbations on categorical features"""
        cat = CategoricalFeatures(self.X_test, feature_index)
        cat_arr = cat.apply()
        self.model_avg_error.append(self.calculate_errors(self.y_test, self.model.predict(cat_arr)))

    def calculate_variance(self):
        """
        a method that calculate the variance.
        """
        self.model_avg_error.append(self.calculate_errors(self.y_test, self.model.predict(self.X_test)))
        for feature_index in range(self.X_test.shape[1]):
            feature = self.X_test[:, feature_index]
            if np.issubdtype(feature.dtype, np.integer) or np.issubdtype(feature.dtype, np.floating):
                self.__get_numerical(feature_index, False)
            else:
                if np.char.isnumeric(feature).all() or np.char.isdecimal(feature).all():
                    self.__get_numerical(feature_index, True)
                else:
                    self.__get_categorical(feature_index)

    def get_diff(self):
        """
        Get the difference between X_test predictions error and each feature prediction error after perturbations.

        return:
         list: difference between X_test predictions error and each feature prediction error that exceeds the threshold,
               with the names of the features instead of indexes.
        """
        exceeds_threshold = []
        feature_names = self.X_test_with_features_name.columns.tolist()  # Convert column names to a list
        for i in range(len(self.model_avg_error[1:])):
            if abs(self.model_avg_error[i] - self.model_avg_error[0]) > self.yaml_reader.get('variance')['threshold']:
                exceeds_threshold.append(feature_names[i])
        return exceeds_threshold

    def __str__(self):
        """
        return:
         str: a summary about detecting high variance prediction or not.
        """
        summary = ''
        features_exceed_threshold = self.get_diff()  # This now contains feature names
        if len(features_exceed_threshold) > 0:
            summary += f'High predictions variance is detected when doing small perturbations on features '
            i = 0
            for column in self.X_test_with_features_name.columns:
                if i in features_exceed_threshold:
                    summary += f'{column}, '
                i += 1
            summary += (f'\nwhich suggests model overfitting on the training set on these specific features '
                        f'or the high importance of these features.')
        else:
            summary = 'High predictions variance is NOT detected when doing small perturbations on features '
        return summary
