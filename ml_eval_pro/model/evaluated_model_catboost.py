import mlflow

from ml_eval_pro.model.evaluated_model import EvaluatedModel


class EvaluatedModelCatBoost(EvaluatedModel):

    def load(self):
        """
        Loading the model from the catboost flavor.
        """
        print(f"Loading the model ...")
        return mlflow.catboost.load_model(model_uri=self.model_uri)

    def predict(self, data, predict_class=True):
        """
        Initializing the evaluation metric needed values.
        :param data: the data to be predicted.
        :param predict_class: indicating whether the prediction is a class prediction (in case of classification only).
        """

        if self.problem_type == "classification" and not predict_class:
            return self.model.predict_proba(data)

        return self.model.predict(data)
