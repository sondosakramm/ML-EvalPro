from ml_eval_pro.carbon.carbon_emission.carbon import Carbon
from ml_eval_pro.carbon.carbon_emission.carbon_calculator import CarbonCalculator
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class CarbonSummary(SummaryGenerator):
    def __init__(self, carbon: Carbon):
        self.carbon = carbon

    def get_summary(self):
        return (f'Current platform is deployed on virtual machine with cpu {self.carbon.cpu_name}, '
                f'it\'s speed is {self.carbon.cpu_speed} and '
                f'power consumptions is {self.carbon.cpu_value} Kw,'
                f', assuming that the energy generator is {self.carbon.energy_generator}, '
                f'and it\'s power consumption is {self.carbon.energy_generator_value} KgCO2 so it will generate'
                f' {CarbonCalculator(self).calculate_carbon()} CO2 for a single prediction, '
                f'{CarbonCalculator(self).calculate_predictions()} predictions will generate 1 Kg Co2')
