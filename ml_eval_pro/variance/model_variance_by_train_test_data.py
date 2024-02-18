from ml_eval_pro.variance.model_var.model_variance import ModelVariance


class ModelVarianceByTrainTestData(ModelVariance):
    """
    A class for measuring the model variance by using train/test data.
    """
    def __init__(self, model, X_test, y_test, X_train, y_train, problem_type='regression', metric='MAE'):
        """
        Initialize a ModelVarianceByTrainTestData instance.

        Parameters:

        model : object, The machine learning model to assess for variance.

        X_test : array-like or pd.DataFrame The feature matrix of the test dataset.

        y_test : array-like or pd.Series The true labels or target values for the test dataset.

        X_train : array-like or pd.DataFrame, The feature matrix of the training dataset.

        y_train : array-like or pd.Series, The true labels or target values for the training dataset.

        problem_type : str, optional (default='regression') The type of problem,
            either 'regression' or 'classification'.

        metric : str, optional (default='MAE') The evaluation metric to be used for calculating errors.
         Defaults to 'MAE' (Mean Absolute Error) for regression problems.

        """
        super().__init__(model, X_test, y_test, X_train, y_train, problem_type, metric)


    def calculate_variance(self):
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_error = self.calculate_errors(self.y_train, train_pred)
        test_error = self.calculate_errors(self.y_test, test_pred)

        self.variance = abs(test_error - train_error)
        return self.variance

    def __str__(self):
        """Return: str
           a summary about detecting high variance or not."""
        summary = f''
        if self.variance > self.yaml_reader.get('variance')['threshold']:
            summary += f'High variance is detected.'
        else:
            summary += f'No high variance is detected.'
        return summary
