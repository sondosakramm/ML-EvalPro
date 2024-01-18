class YamlSerializer:
    """
    A serializer for extracting specific fields from a YAML file and returning them as a dictionary.

    """
    def __init__(self, yaml_file):
        """
        Initialize a YamlSerializer instance.

        Parameters:
        - yaml_file: The YAML file containing data to be serialized.

        """
        self.file = yaml_file

    def __dict__(self):
        """
        Serialize the YAML data into a dictionary.

        Returns:
        - dict: A dictionary containing selected fields from the YAML data.

        """

        return {
            'energy_generator': self.file['energy_generator'],
            'cpu': self.file['cpu'],
            'variance': self.file['variance']
        }