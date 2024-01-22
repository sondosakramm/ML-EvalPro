import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

from auto_evaluator.adverserial_test_cases.adversarial_attack import AdversarialAttack


class AdversarialAttributeInferenceAttack(AdversarialAttack):
    """
    A class for generating an attribute inference adversarial attack.
    """
    def generate(self) -> np.ndarray:
        inferred_bb = np.array([])

        model_type = self._get_model_type()
        art_model = AdversarialAttributeInferenceAttack._generate_adversarial_attack_model(model_type[0])

        for attack_feature in range(self.input_features.shape[1]):
            attack_predictions = np.array(
                [np.argmax(arr) for arr in art_model.predict(self.input_features)]).reshape(-1, 1)

            attack_feature_removed = np.delete(self.input_features, attack_feature, 1)

            bb_attack = AttributeInferenceBlackBox(art_model, attack_feature=attack_feature, attack_model_type=model_type[1])
            bb_attack.fit(self.input_features)

            inferred_feature_bb = bb_attack.infer(attack_feature_removed, pred=attack_predictions)

            # TODO: Need to consider the correlation among the features
            inferred_bb = np.concatenate((inferred_bb, inferred_feature_bb), axis=1)

        return inferred_bb