from ml_eval_pro.adverserial_test_cases.adversarial_attack_factory import AdversarialAttackFactory
from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator


class AdversarialEvaluator(BaseEvaluator):
    """
    A class of the adversarial evaluator.
    """
    def __init__(self, evaluator: InterfaceEvaluator):
        """
        Evaluate the adversarial attack from the different evaluation analysis provided.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)
        self.__adversarial_testcases = None
        self.__robust = None

    def evaluate(self, **kwargs):
        """
        Evaluate the adversarial attack from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the adversarial attacks.
        """
        super().evaluate(**kwargs)

        print("Generating the model adversarial test cases ...")
        dataset_columns = list(self.test_dataset.columns)

        adversarial_attack_model = AdversarialAttackFactory.create("substitute_zoo",
                                                                   self.model_pipeline, self.problem_type,
                                                                   self.test_dataset, self.test_target,
                                                                   dataset_columns,
                                                                   train_input_features=self.train_dataset,
                                                                   train_target_features=self.train_target,
                                                                   num_classes=self.num_classes)

        self.__adversarial_testcases = adversarial_attack_model.get_adversarial_testcases()
        self.__robust = adversarial_attack_model.is_robust

    @property
    def adversarial_testcases(self):
        return self.__adversarial_testcases

    @property
    def robust(self):
        return self.__robust
