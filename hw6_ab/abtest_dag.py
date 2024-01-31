import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F

# from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
# from pyspark.ml.feature import VectorAssembler
from pyspark import SparkConf, SparkContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from airflow.decorators import dag, task
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import mannwhitneyu


def f1_score(predictions):
    '''Подсчёто метрики F1 для предсказаний модели
       на вохд предсказания с колонками tx_fraud - лейблы, prediction - ответы модели
    '''
    tp = predictions.filter((F.col("tx_fraud") == 1) & (F.col("prediction") == 1)).count()
    tn = predictions.filter((F.col("tx_fraud") == 0) & (F.col("prediction") == 0)).count()
    fp = predictions.filter((F.col("tx_fraud") == 0) & (F.col("prediction") == 1)).count()
    fn = predictions.filter((F.col("tx_fraud") == 1) & (F.col("prediction") == 0)).count()
    
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = 2*(prec*rec)/(prec+rec)
    return f1

def get_spark_session():
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession
    spark = (
        SparkSession
        .builder
        .appName("antiFraud")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )
    return spark

default_args = {
    'owner': 'aurflow',
    'depends_on_past': False,

    'retries': 1, # Кол-во попыток выполнить DAG
    'retry_delay': timedelta(minutes=3),

    'start_date': datetime(2023, 1, 1), # Дата начала выполнения DAG
}

schedule_interval = '0 12 * * *' # cron-выражение, также можно использовать '@daily', '@weekly', а также timedelta


@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def validate_models():
    
    @task
    def get_predictions():
        spark = get_spark_session()
        df = spark.read.parquet("/user/data_clean/df.parquet").fillna(0)

        modelA = PipelineModel.load("/user/models/tree/sparkml") #/user/models/tree_md7/sparkml
        modelB = PipelineModel.load("/user/models/tree_md7/sparkml")

        splits = df.randomSplit([0.9, 0.1], 42)
        training = splits[0]
        test = splits[1]      
        
        predictionsA = modelA.transform(test).select("transaction_id", "tx_fraud", "prediction")
        predictionsB = modelB.transform(test).select("transaction_id", "tx_fraud", "prediction")

        return predictionsA, predictionsB

    @task
    def bootstrap(predictionsA, predictionsB, bootstrap_size = 50, sample_size = 0.8):
        f1A = []
        f1B = []

        for i in range(bootstrap_size):
            sampleA = predictionsA.sample(withReplacement=True, fraction=sample_size)
            f1A.append(f1_score(sampleA))
            
            sampleB = predictionsB.sample(withReplacement=True, fraction=sample_size)
            f1B.append(f1_score(sampleB))
        
        return np.array(f1A), np.array(f1B) 


    @task
    def tests_mw(f1A, f1B):

        stat, p = mannwhitneyu(f1A, f1B)
        intA = ((f1A.mean() - 3*f1A.std()), (f1A.mean() + 3*f1A.std()))
        intB = ((f1B.mean() - 3*f1B.std()), (f1B.mean() + 3*f1B.std()))
        
        return stat, p, intA, intB

    
    @task
    def result(stat, p, intA, intB, alpha = 0.05):
        print(stat, p, intA, intB)
        
        if p > alpha:
            print('Одинаковые распределения (не отвергаем H0)')
        else:
            print('Разные распределения (отвергаем H0)')
            
        if intA[1] < intB[0]:
            print('Модель В лучше')
        elif intB[1] < intA[0]:
            print('Модель A лучше')
        else:
            print('Нельзя предположить какая модель лучше')
    
    predictionsA, predictionsB = get_predictions()
    f1A, f1B = bootstrap(predictionsA, predictionsB)
    stat, p, intA, intB = tests_mw(f1A, f1B)
    result = result(stat, p, intA, intB) 
    
validate_models = validate_models()
