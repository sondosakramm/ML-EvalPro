import mlflow

from ml_eval_pro.model.evaluated_model import EvaluatedModel


class EvaluatedModelSKLearn(EvaluatedModel):

    def load(self):
        print(f"Loading a sklearn model ...")
        return mlflow.sklearn.load_model(model_uri=self.model_uri)

    def predict(self, data, predict_class=True):
        if self.problem_type == "classification" and not predict_class:
            return self.model.predict_proba(data)

        return self.model.predict(data)

