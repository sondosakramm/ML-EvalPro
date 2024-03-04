from ml_eval_pro.carbon.inference_time.inference_time import InferenceTime
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class InferenceSummary(SummaryGenerator):
    def __init__(self, inference: InferenceTime):
        self.inference = inference

    def get_summary(self):
        return (f'The model average inference time per instance is {str(self.inference.calc_inference_time_seconds())}'
                f' seconds')
