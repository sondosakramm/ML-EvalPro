from abc import ABC, abstractmethod


class AdversarialAttack(ABC):
    """
    An abstract class for generating an adversarial attack.
    """
    def __init__(self):
        self.is_robust = None

    @abstractmethod
    def get_adversarial_testcases(self):
        """
        Evaluate the model predictions and the adversarial test cases.
        :return: the adversarial test cases after evaluation.
        """
        pass
