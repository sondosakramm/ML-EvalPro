import yaml
from ml_eval_pro.configuration_manager.configuration_reader.configuration_reader_interface import ConfigurationReader
from ml_eval_pro.configuration_manager.readers_serializers.yaml_serializer import YamlSerializer


class YamlReader(ConfigurationReader):
    """
    A configuration reader for reading configuration data from YAML files.

    """

    def __init__(self, file_path: str):
        """
        Initialize a YamlReader instance.

        Parameters:
        - file_path: The file path for the YAML file.

        """
        self.file = self.open(file_path)
        self.serializer = YamlSerializer(self.file)

    def open(self, file_path: str):
        """
        Open and read a YAML file.

        Parameters:
        - file_path: The file path for the YAML file.

        Returns:
        - dict: A dictionary containing the YAML data.

        """

        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        return yaml_data

    def get(self, config_var_name: str):
        """
        Get a specific configuration variable from the YAML data.

        Parameters:
        - config_var_name: The name of the configuration variable.

        Returns:
        - Any: The value of the specified configuration variable from the YAML data.

        """
        return self.serializer.__dict__()[config_var_name]
