import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from scipy.stats import norm
from auto_evaluator.evaluation_metrics.regression.regression_evaluator import RegressionEvaluator


class Calibration(RegressionEvaluator):
    """
    Calibration of regression model.
    """

    def __init__(self, target, prediction, bins: int = 10):
        super().__init__(target, prediction)
        self.bins = bins


    def measure(self):
        global max_actual
        min_value = self.prediction.min()
        max_value = self.prediction.max()
        predictions_binning_ranges = np.linspace(min_value, max_value, self.bins + 1)

        actual_binning_ranges = np.array([])

        for i in range(1, self.bins + 1):
            # Getting the values of the actual target values in the current bin range.
            actual_bin_range = self.target[
                np.logical_and(self.target >= predictions_binning_ranges[i - 1], self.target < predictions_binning_ranges[i])]

            # Getting the minimum value of the current bin from the maximum of the previous bin to get a proper line
            min_actual = self.target.min() if i == 1 else max_actual
            max_actual = max(actual_bin_range)

            actual_binning_ranges = np.append(actual_binning_ranges, min_actual)

            if i == self.bins:
                actual_binning_ranges = np.append(actual_binning_ranges, self.target.max())

        return predictions_binning_ranges, actual_binning_ranges

    def display(self):
        predictions_binning_ranges, actual_binning_ranges = self.measure()
        self.target.sort()
        self.prediction.sort()
        # Plot the graph
        plt.plot(actual_binning_ranges, actual_binning_ranges, linestyle='--', color='grey',
                 label='Perfect Calibrartion')
        plt.plot(self.prediction, self.target, linestyle='-', marker="o", color='r', label='Dataset Points')

        # Set x-axis and y-axis limits
        plt.xlim(min(predictions_binning_ranges), max(predictions_binning_ranges))
        plt.ylim(min(actual_binning_ranges), max(actual_binning_ranges))

        # Add labels and title
        plt.xlabel('Binned Predictions')
        plt.ylabel('Max Actual/ Min Actual for each Bin')
        plt.title('Calibration Plot for Regression')

        # Add a legend
        plt.legend()

        # Display the graph
        plt.show()
