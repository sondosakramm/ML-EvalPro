from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelUnlearning(GdprCompliance):
    # Data Deletion (Anti-sample Generation/Fisher forgetting/Zero shot/Incompetent teacher)
    def __str__(self):
        raise NotImplementedError
