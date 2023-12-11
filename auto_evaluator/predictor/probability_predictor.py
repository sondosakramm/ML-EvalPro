from auto_evaluator.predictor.predictor import Predictor


class ProbabilityPredictor(Predictor):
    def predict_model(self):
        """
        Predicting the features target probability values of a model.
        :return: the prediction probability values of a model.
        """
        return self.model_pipeline.predict_proba(self.features)
