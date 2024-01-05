from abc import ABC, abstractmethod

class FeatureImportance(ABC):
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    @abstractmethod
    def calculate(self):
        pass
