from ml_eval_pro.carbon.inference_time.inference_time import InferenceTime


class Carbon:
    """
    A class for carbon emission.

    """

    def __init__(self, model, data, cpu_name: str = 'Xeon', cpu_speed: float = 2.2, cpu_value: float = 0.061,
                 energy_name: str = 'Fuel', energy_value: float = 0.865):
        """
        Initialize a Carbon instance.

        Parameters:
        - model: model that it's inference time will be calculated.
        - data: (train/test) data that model will use to generate predictions.
        - cpu_name (str): The name of the CPU being used.
        - cpu_speed (float): The speed of the CPU in GHz.
        - cpu_value (float): The value or performance index of the CPU.
        - energy_name (str): The name or type of energy resource being used.
        - energy_value (float): The value or consumption rate of the energy resource.
        """
        self._inference_time = InferenceTime(model, data).calc_inference_time_hours()
        self._cpu_name = cpu_name
        self._cpu_speed = cpu_speed
        self._cpu_value = cpu_value
        self._energy_generator = energy_name
        self._energy_generator_value = energy_value

    @property
    def cpu_name(self):
        if hasattr(self, '_cpu_name'):
            return self._cpu_name
        else:
            raise Exception(f'ERROR: No attribute called cpu_name')

    @property
    def inference_time(self):
        if hasattr(self, '_inference_time'):
            return self._inference_time
        else:
            raise Exception(f'ERROR: No attribute called inference time')

    @property
    def energy_generator(self):
        if hasattr(self, '_energy_generator'):
            return self._energy_generator
        else:
            raise Exception(f'ERROR: No attribute called energy_generator')

    @property
    def cpu_value(self):
        if hasattr(self, '_cpu_value'):
            return self._cpu_value
        else:
            raise Exception(f'ERROR: No attribute called cpu_value')

    @property
    def cpu_speed(self):
        if hasattr(self, '_cpu_speed'):
            return self._cpu_speed
        else:
            raise Exception(f'ERROR: No attribute called cpu_speed')

    @property
    def energy_generator_value(self):
        if hasattr(self, '_energy_generator_value'):
            return self._energy_generator_value
        else:
            raise Exception(f'ERROR: No attribute called energy_generator_value')
