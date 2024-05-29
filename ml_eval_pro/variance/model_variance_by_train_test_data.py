from ml_eval_pro.variance.model_var.model_variance import ModelVariance


class ModelVarianceByTrainTestData(ModelVariance):
    """
    A class for measuring the model variance by using train/test data.
    """
    def __init__(self, model, model_type, test_dataset, target, train_dataset, train_target, evaluation_metric,
                 threshold: float):
        """
        Initialize a ModelVarianceByTrainTestData instance.

        Parameters:

        model : object, The machine learning model to assess for variance.

        test_dataset : array-like or pd.DataFrame The feature matrix of the test dataset.

        target : array-like or pd.Series The true labels or target values for the test dataset.

        train_dataset : array-like or pd.DataFrame, The feature matrix of the training dataset.

        train_target : array-like or pd.Series, The true labels or target values for the training dataset.

        model_type : str, optional (default='regression') The type of problem,
            either 'regression' or 'classification'.

        evaluation_metric : str, optional (default='MAE') The evaluation metric to be used for calculating errors.
         Defaults to 'MAE' (Mean Absolute Error) for regression problems.

        """
        self.train_target = train_target
        self.train_dataset = train_dataset
        self.threshold = threshold
        super().__init__(model=model, test_dataset=test_dataset, target=target, model_type=model_type,
                         evaluation_metric=evaluation_metric)


    def calculate_variance(self):
        train_pred = self.model.predict(self.train_dataset)
        test_pred = self.model.predict(self.test_dataset)

        train_error = self.calculate_errors(self.train_target, train_pred)
        test_error = self.calculate_errors(self.target, test_pred)

        variance = abs(test_error - train_error)
        return variance
