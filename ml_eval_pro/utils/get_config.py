def get_config(**kwargs):
    config = {
        'shap_threshold': 0.05,
        'ece_threshold': 0.05,
        'bias_threshold': 0.15,
        'variance_threshold': 0.15,
        'variance_step_size': 0.5,
        'cpu_name': 'Xeon',
        'cpu_speed': 2.2,
        'cpu_value': 0.061,
        'energy_name': 'Fuel',
        'energy_value': 0.865,
        'llama_model': 'llama3.2',
        'bins': 10
    }

    for config_key in kwargs.keys():
        config[config_key] = kwargs[config_key]

    return config
