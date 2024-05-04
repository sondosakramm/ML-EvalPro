import time
from decimal import getcontext
from decimal import Decimal


class InferenceTime:
    """
    A class for calculating model inference time.

    """

    def __init__(self, model, data):
        """
        Initialize an InferenceTime instance.

        Parameters:
        - model: model that it's inference time will be calculated.
        - data: (train/test) data that model will use to generate predictions.

        """
        self.model = model
        self.data = data

    def calc_inference_time_seconds(self):
        """
        Calculating inference time in seconds.

        Return:
        - float: inference time in seconds.

        """
        getcontext().prec = 20

        start_time = time.perf_counter_ns()
        self.model.predict(self.data)
        end_time = time.perf_counter_ns()

        sec = Decimal(end_time - start_time) / Decimal(1e9)

        if sec > 0:
            seconds = Decimal(sec) / Decimal(len(self.data))
            return seconds
        else:
            return -1

    def calc_inference_time_hours(self):
        """
        Calculating inference time in hours.

        Return:
        - float: inference time in hours.

        """
        getcontext().prec = 20
        hours = self.calc_inference_time_seconds() / Decimal(3600)
        return hours

    def calc_inference_time_minutes(self):
        """
        Calculating inference time in minutes.

        Return:
        - float: inference time in minutes.

        """
        getcontext().prec = 20
        minutes = Decimal(self.calc_inference_time_seconds() / Decimal(60))
        return minutes

    def __str__(self):
        """
        Return:
        - str: a string of the inference time.
        """
        msg = f'The model average inference time per instance is {self.calc_inference_time_seconds():.10f} seconds'
        return msg
