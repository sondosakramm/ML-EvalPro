
class CarbonCalculator:
    """
    A class for calculating carbon emission.

    """
    def __init__(self, carbon):
        """
        Initialize a CarbonCalculator instance.

        Parameters:
        - carbon: Carbon object that contain all the carbon/system info.

        """
        self.carbon = carbon


    def calculate_carbon(self):
        """
        Calculate carbon emissions.

        return:
        - float: carbon emissions per prediction.

        """
        carbon_per_prediction = self.carbon.energy_generator_value * self.carbon.cpu_value * self.carbon.inference_time
        return carbon_per_prediction

    def calculate_predictions(self):
        """
        Calculate carbon emissions per prediction.

        return:
        - int: number of predictions that will generate 1Kg CO2.

        """
        predictions_needed = 1 / self.calculate_carbon()
        return round(predictions_needed)

