import mlflow
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from ml_eval_pro.model.evaluated_model import EvaluatedModel


class EvaluatedModelSparkMLLib(EvaluatedModel):
    """
    A class for generating the evaluated spark model object.
    """

    def __init__(self, model_uri, model_type, problem_type, spark_feature_col_name, spark_session_name):
        """
        Initializing the evaluation metric needed values.
        :param model_uri: the model uri.
        :param model_type: the model type (flavor).
        :param problem_type: the problem type (regression or classification).
        :param spark_feature_col_name: the spark features column name (used in spark models only).
        :param spark_session_name: the spark name session (used in spark models only).
        """
        super().__init__(model_uri, model_type, problem_type)
        self.spark_feature_col_name = spark_feature_col_name
        self.spark_session = spark = SparkSession.builder.appName(spark_session_name).getOrCreate()

    def load(self):
        """
        Loading the model from the spark flavor.
        """
        print(f"Loading the model ...")
        return mlflow.spark.load_model(model_uri=self.model_uri)

    def predict(self, data, predict_class=True):
        """
        Initializing the evaluation metric needed values.
        :param data: the data to be predicted.
        :param predict_class: indicating whether the prediction is a class prediction (in case of classification only).
        """
        df_spark = self.spark_session.createDataFrame(data)

        assembler = VectorAssembler(inputCols=data.columns.tolist(), outputCol=self.spark_feature_col_name)
        transformed_data = assembler.transform(df_spark)

        predictions = self.model.transform(transformed_data).toPandas()

        if self.problem_type == "classification":
            if predict_class:
                return predictions["prediction"].to_numpy()
            else:
                predictions_probability = predictions["probability"].to_numpy()
                predictions_probability_1 = np.array([p.toArray().tolist() for p in predictions_probability])
                return predictions_probability_1

        elif self.problem_type == "regression":
            return predictions["prediction"].to_numpy()

        return predictions
