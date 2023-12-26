import time
from decimal import getcontext


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
        start_time = time.time()
        self.model.predict(self.data)
        end_time = time.time()
        getcontext().prec = 10
        seconds = (end_time - start_time) / len(self.data)
        return seconds

    def calc_inference_time_hours(self):
        """
        Calculating inference time in hours.

        Return:
        - float: inference time in hours.

        """
        getcontext().prec = 10
        hours = self.calc_inference_time_seconds() / 3600
        return hours

    def calc_inference_time_minutes(self):
        """
        Calculating inference time in minutes.

        Return:
        - float: inference time in minutes.

        """
        getcontext().prec = 10
        minutes = self.calc_inference_time_seconds() / 60
        return minutes

    def __str__(self):
        """
        Return:
        - str: a string of the inference time.
        """
        msg = (f'Model inference time was {str(self.calc_inference_time_hours())} hours, '
               f'{self.calc_inference_time_minutes()} minutes, '
               f'{self.calc_inference_time_seconds()} seconds')
        return msg
