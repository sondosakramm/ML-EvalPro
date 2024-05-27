from ml_eval_pro.transparency.transparency import Transparency


class TransparencyCatBoost(Transparency):
    """
    A class for generating transparency for catboost models.
    """

    def get_model_algorithm(self):
        pass

    def get_model_score(self, model_algorithm, **kwargs):
        return 110
