import numpy as np
from art.attacks.evasion.zoo import ZooAttack
from art.estimators import KerasEstimator
from art.estimators.classification import SklearnClassifier, TensorFlowClassifier, TensorFlowV2Classifier, \
    KerasClassifier

from art.estimators.regression import KerasRegressor
import tensorflow as tf
from keras import Input, Model
from sklearn.metrics import f1_score

tf.compat.v1.disable_eager_execution()
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# from keras.models import Sequential
# from keras.layers import Dense

from keras.losses import mean_squared_error

from auto_evaluator.adverserial_test_cases.adversarial_attack import AdversarialAttack
from auto_evaluator.evaluation_metrics.evaluators_factory import EvaluatorsFactory


class AdversarialEvasionAttack(AdversarialAttack):
    def __init__(self, model, model_type: str, input_features: np.ndarray, target_features: np.ndarray,
                 significance=0.05, num_classes=None):
        """
        Create an evasion adversarial attack.

        The attack starts with training a classification/neural network model. Then, the model is utilized in a ZOO attack,
        which only accepts a gradient-based models. The results from:
        - The blackbox original model predictions of the test cases, and
        - The blackbox original model predictions of the adversarial test cases
         are evaluated to assure the model robustness to adversarial attacks.

        :param model: the model to be tested against adversarial attacks.
        :param model_type: the model type i.e. Regression, Classification, etc.
        :param input_features: the input features of the dataset.
        :param target_features: the target feature(s) of the dataset.
        :param num_classes: the number of classes of a classification model.
        :param significance: the significance value of the evaluation to the defined threshold.
        :return: the created adversarial attack.
        """
        super().__init__(model, model_type, input_features, target_features)
        self.model_predictions = self.model.predict(self.input_features)
        self.significance = significance
        self.num_classes = num_classes


    def generate(self) -> np.ndarray:
        """
        Generate adversarial test cases.
        :return: the generated adversarial test cases.
        """
        art_model = None
        surrogate_model = None
        attack_model = None

        if self.model_type == 'classification':
            surrogate_model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.input_features.shape[1],)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])

            surrogate_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            tf.keras.backend.set_value(surrogate_model.optimizer.learning_rate, 0.01)

            surrogate_model.fit(self.input_features, self.model_predictions, epochs=100)

            art_model = KerasClassifier(model=surrogate_model)

            attack_model = ZooAttack(art_model, nb_parallel=self.num_classes, learning_rate=0.01, max_iter=20)

        elif self.model_type == 'regression':
            surrogate_model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.input_features.shape[1],)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])

            surrogate_model.compile(optimizer="adam",
                          loss='mean_squared_error',
                          metrics=['mse'])

            tf.keras.backend.set_value(surrogate_model.optimizer.learning_rate, 0.01)

            surrogate_model.fit(self.input_features, self.model_predictions, epochs=100)

            for layer in surrogate_model.layers:
                layer.trainable = False

            # art_model = KerasRegressor(model=surrogate_model, clip_values=(0,1))
            # Discretize the regression predictions into classes (you might need to adjust the bins)
            num_bins = 100 if self.model_predictions.shape[0] > 100 else int(self.model_predictions.shape[0] / 2)
            labels = np.digitize(self.model_predictions,
                                          bins=np.linspace(min(self.model_predictions), max(self.model_predictions), num_bins))

            labels = np.clip(labels, 1, num_bins)

            # # Convert to one-hot encoding (classification)
            # labels_one_hot = tf.keras.utils.to_categorical(labels - 1, num_classes=num_bins)
            encoded_arr = np.zeros((labels.size, labels.max()), dtype=int)
            encoded_arr[np.arange(labels.size), labels-1] = 1
            print(encoded_arr[0])
            # Build a classification model
            surrogate_model_classification = tf.keras.models.Sequential([
                surrogate_model,
                # tf.keras.layers.InputLayer(input_shape=(self.input_features.shape[1],)),
                # tf.keras.layers.Dense(128, activation='relu'),
                # tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_bins, activation='softmax')  # Adjust to match the number of bins
            ])

            surrogate_model_classification.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])

            tf.keras.backend.set_value(surrogate_model_classification.optimizer.learning_rate, 0.01)

            # Train the classification model
            surrogate_model_classification.fit(self.input_features, encoded_arr,
                                epochs=100)  # You may need to adjust the number of epochs

            # Wrap the classification model using KerasRegressor
            art_model = KerasClassifier(model=surrogate_model_classification)

            # attack_model = ZooAttack(art_model, nb_parallel=5)
            # adversarial_samples = attack_model.generate(self.input_features)

            # 1st Method
            # adversarial_labels_one_hot = surrogate_model.predict(adversarial_samples).flatten()[:len(self.model_predictions)]
            #
            # # Convert one-hot encoded labels to class labels
            # adversarial_labels = np.argmax(adversarial_labels_one_hot, axis=1)
            #
            # # Find the center of the bin for each adversarial label
            # bin_centers = np.linspace(min(self.model_predictions), max(self.model_predictions), num_bins)
            # adversarial_predictions = bin_centers[adversarial_labels]

            # 2nd Method
            # adversarial_labels_one_hot = surrogate_model.predict(adversarial_samples)
            #
            # # Convert one-hot encoded labels to class labels
            # adversarial_labels = np.argmax(adversarial_labels_one_hot, axis=1)
            #
            # # Compute the mean of each bin as the representative value
            # bin_means = np.zeros(num_bins)
            # for i in range(1, num_bins + 1):
            #     bin_indices = np.where(labels == i)
            #     bin_means[i - 1] = np.mean(self.model_predictions[bin_indices])
            #
            # # Map the adversarial labels to the mean of each bin
            # adversarial_predictions = bin_means[adversarial_labels]

            # 3rd Method
            # adversarial_labels_one_hot = surrogate_model.predict(adversarial_samples)
            #
            # # Compute the weighted average of the bin boundaries using predicted probabilities
            # bin_boundaries = np.linspace(min(self.model_predictions), max(self.model_predictions), num_bins + 1)
            # adversarial_predictions = np.zeros(adversarial_labels_one_hot.shape[0])
            #
            # for i in range(adversarial_labels_one_hot.shape[0]):
            #     adversarial_predictions[i] = np.average(bin_boundaries[:-1], weights=adversarial_labels_one_hot[i])
            #
            # print(self.model_predictions)
            # print(adversarial_predictions)
            #
            # eval_metric = EvaluatorsFactory.get_evaluator("mape", self.model_predictions,
            #                                               adversarial_predictions).measure()
            #
            # return abs(eval_metric - self.threshold) <= self.significance

            # Initialize and perform the Zoo attack
            # attack_model = ZooAttack(art_model)
            # adversarial_samples = attack_model.generate(self.input_features)

            attack_model = ZooAttack(art_model, nb_parallel=5, learning_rate=0.01, max_iter=20)

        return attack_model.generate(self.input_features)

    def evaluate_robustness(self, adversarial_examples):
        """
        Evaluate the model robustness to adversarial attacks.
        :param adversarial_examples: the generated adversarial examples.
        :return: a flag indicating whether the model is robust to adversarial attacks.
        """
        adversarial_predictions = self.model.predict(adversarial_examples)
        eval_metric = 0

        if self.model_type == "regression":
            eval_metric = EvaluatorsFactory.get_evaluator("mape", self.model_predictions,
                                                          adversarial_predictions).measure()
        elif self.model_type == "classification":
            eval_metric = EvaluatorsFactory.get_evaluator("cross entropy loss", self.model_predictions,
                                                          adversarial_predictions).measure()

        print(eval_metric)
        return eval_metric <= self.significance

    def __str__(self):
        model_robust = self.evaluate_robustness(self.generate())

        robust_str = "robust" if model_robust else "not robust"

        return (f'According to the results of evaluating the model predictions of the original examples'
                f' to the adversarial examples, the model is {robust_str} to adversarial attacks.')
