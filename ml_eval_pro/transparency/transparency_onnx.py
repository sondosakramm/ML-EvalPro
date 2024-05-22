from typing import Tuple, List

import onnx

from ml_eval_pro.transparency.transparency import Transparency


class TransparencyONNX(Transparency):
    def get_model_algorithm(self):
        op_types = [node.op_type for node in onnx.load(self.model.model._model_impl.rt._model_path).graph.node]
        if len(op_types) == 1:
            return op_types[0]
        return None

    def get_model_algorithms_complexity(self) -> Tuple[List[str], List[str], List[str]]:
        explainable_models = ['LinearRegressor', 'LinearClassifier', 'DecisionTreeRegressor', 'DecisionTreeClassifier']
        partially_explainable_models = ['SVMRegressor', 'SVMClassifier']
        complex_models = ['DNN', 'CNN', 'RNN', 'LSTM', 'GRU', 'Transformer']

        return explainable_models, partially_explainable_models, complex_models

    def get_model_score(self, model_algorithm, **kwargs):
        explainable_models = kwargs["explainable_models"]
        partially_explainable_models = kwargs["partially_explainable_models"]
        complex_models = kwargs["complex_models"]

        if model_algorithm is None:
            return 110

        else:
            if model_algorithm in explainable_models:
                return 25
            elif model_algorithm in partially_explainable_models:
                return 65
            elif model_algorithm in complex_models:
                return 110

        return -1
