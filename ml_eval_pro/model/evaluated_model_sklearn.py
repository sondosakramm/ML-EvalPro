from ml_eval_pro.model.evaluated_model import EvaluatedModel


class EvaluatedModelSKLearn(EvaluatedModel):

    def predict(self, data):
        if self.problem_type == "regression":
            return self.model.predict(data)
        else:
            return self.model.predict_proba(data)

    def predict_class(self, data):
        if self.problem_type == "regression":
            return self.model.predict(data)
        else:
            raise TypeError("This method is not supported for classification problems!")
