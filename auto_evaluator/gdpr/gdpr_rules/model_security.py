from auto_evaluator.gdpr.gdpr_compliance import GdprCompliance


class ModelSecurity(GdprCompliance):

    def __get_adversarial_test_cases(self):
        pass

    def __evaluate(self):
        # Get X_test predictions

        # Get adversarial test cases predictions

        # Evaluate X_test and y_test (1)

        # Evaluate adversarial test cases and y_test (2)

        # Get difference between (1) & (2), compare to threshold

        # Large difference = Not Secure

        # Low difference = Secure
        pass

    def __str__(self):
        pass
