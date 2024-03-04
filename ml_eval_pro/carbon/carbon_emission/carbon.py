import os.path

from ml_eval_pro.carbon.inference_time.inference_time import InferenceTime
from ml_eval_pro.configuration_manager.configuration_reader.yaml_reader import YamlReader


class Carbon:
    """
    A class for carbon emission.

    """

    def __init__(self, model, data):
        """
        Initialize a Carbon instance.

        Parameters:
        - model: model that it's inference time will be calculated.
        - data: (train/test) data that model will use to generate predictions.

        """
        self.__yaml_reader = YamlReader(os.path.join(os.path.curdir, "ml_eval_pro",
                                                     "config_files", "system_config.yaml"))
        self._inference_time = InferenceTime(model, data).calc_inference_time_hours()
        self._cpu_name = self.__yaml_reader.get('cpu')['name']
        self._cpu_speed = self.__yaml_reader.get('cpu')['speed']
        self._cpu_value = self.__yaml_reader.get('cpu')['value']
        self._energy_generator = self.__yaml_reader.get('energy_generator')['name']
        self._energy_generator_value = self.__yaml_reader.get('energy_generator')['value']

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
