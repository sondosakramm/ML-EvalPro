from auto_evaluator.predictor.predictor import Predictor

class NormalPredictor(Predictor):
    def predict_model(self):
        """
        Predicting the features target values of a model.
        :return: the prediction values of a model.
        """
        return self.model_pipeline.predict(self.features)
