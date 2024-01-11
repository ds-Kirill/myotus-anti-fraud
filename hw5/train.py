#!/usr/bin/env python
# coding: utf-8

import findspark
findspark.init()

import fire
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
#from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import MinMaxScaler
from pyspark import SparkConf, SparkContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import mlflow
from mlflow.tracking import MlflowClient

import os

def get_pipeline():
    assembler = VectorAssembler(inputCols=['customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds'], outputCol='features')
    minmaxscaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
    rf = RandomForestClassifier(labelCol="tx_fraud", featuresCol="scaledFeatures", 
                                numTrees=200, 
                                maxDepth=10, 
                                minInstancesPerNode=7,
                                minInfoGain=4,
                                featureSubsetStrategy="sqrt",
                                bootstrap=True,
                                seed=42,  )
    pipeline = Pipeline(stages=[assembler, minmaxscaler, rf])
    return pipeline
    
def main(random_state: int = 42):

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net/'
    os.environ['MLFLOW_TRACKING_URI'] = 'http://158.160.97.198:8000' 

    conf = SparkConf()
    conf.set("spark.executor.memoryOverhead", "4096")  # Увеличьте значение памяти надежности исполнителя
    conf.set("spark.yarn.executor.memoryOverhead", "4096")  # Увеличьте значение памяти надежности исполнителя для YARN
    conf.set("spark.yarn.scheduler.maximum-allocation-mb", "32768")  # Увеличьте максимальное выделение памяти для исполнителей
    conf.set("spark.yarn.nodemanager.resource.memory-mb", "32768")  # Увеличьте общее количество памяти на датаноде
    
    sc = SparkContext(conf=conf)


    spark = (
        SparkSession
        .builder
        .appName("antiFraud")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.cores", "2") 
        .config("spark.executor.cores", "8")  # Установите значение равным количеству ядер на датаноде (8 в вашем случае)
        .config("spark.executor.memory", "28g")  # Установите значение, учитывая доступную память на датаноде и резервируемую память
        .config("spark.executor.instances", "3")  # Установите значение, учитывая доступные ресурсы на кластере
        #.config("spark.default.parallelism", "24")  # Установите значение равным количеству доступных ядер на кластере (24 в вашем случае)
        .config("spark.shuffle.service.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )

    struct = StructType([
    StructField("transaction_id", IntegerType(), nullable=True),
    StructField("tx_datetime", StringType(), nullable=True),
    StructField("customer_id", IntegerType(), nullable=True),
    StructField("terminal_id", IntegerType(), nullable=True),
    StructField("tx_amount", DoubleType(), nullable=True),
    StructField("tx_time_seconds", IntegerType(), nullable=True),
    StructField("tx_time_days", IntegerType(), nullable=True),
    StructField("tx_fraud", IntegerType(), nullable=True),
    StructField("tx_fraud_scenario", IntegerType(), nullable=True)
    ])


    df = spark.read.parquet("/user/data_clean/df.parquet")

    mlflow.set_experiment(experiment_name="Train_RF_Airflow")

    client = MlflowClient()
    experiment = client.get_experiment_by_name("Train_RF_Airflow")
    experiment_id = experiment.experiment_id

    run_name = 'RF_Airflow' + '_' + str(datetime.now())
    FEATURES = ['customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds']

    splits = df.randomSplit([0.95, 0.05], random_state)
    training = splits[0]
    test = splits[1]

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.log_param('features', FEATURES)
        mlflow.log_param('numTrees', 200)
        mlflow.log_param('maxDepth', 10)
        mlflow.log_param('minInstancesPerNode', 7)
        mlflow.log_param('minInfoGain', 4)
        mlflow.log_param('featureSubsetStrategy', "sqrt")
        mlflow.log_param('bootstrap', "bootstrap")
                                
        inf_pipeline = get_pipeline()

        model = inf_pipeline.fit(training)
        
        evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR', labelCol='tx_fraud')
        predictions = model.transform(test)
        areaUnderPR = evaluator.evaluate(predictions)
        
        mlflow.log_metric("areaUnderPR", areaUnderPR)
        mlflow.spark.save_model(model, "RandomForest")

        mlflow.spark.log_model(model, "RandomForest")

    spark.stop()
    

if __name__ == "__main__":
    fire.Fire(main)
