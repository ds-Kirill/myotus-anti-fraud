import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
import argparse

def main(args):
    data_path = '/user/data'
    spark = (
        SparkSession
        .builder
        .appName("antiFraud")
        .config("spark.executor.cores", "4")
        .config("spark.executor.memory", "4g") 
        .config("spark.executor.instances", "6")
        .config("spark.default.parallelism", "48")
        .config("spark.shuffle.service.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "true")
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
    
    df = spark.read\
    .option("header", "true")\
    .option("inferSchema", "false")\
    .option("delimiter", ",")\
    .schema(struct)\
    .csv(data_path)
    
    file_path_parq = data_path + "/" + args + ".parquet"

    df.write.parquet(file_path_parq)
    
    df = spark.read.parquet(file_path_parq)
    
    cleaned_df = clean_data(df)

    # Сохранение очищенных данных
    clean_path = data_path + "_clean/" + args + ".parquet"
    cleaned_df.write.parquet(clean_path)

    # Закрытие SparkSession
    spark.stop()

def clean_data(df):
    # Функции для очистки данных
    
    # 1 -убрать дубли
    df = df.dropDuplicates() 
    
    # 2 - убрать пропуски, в том числе customer_id = -9999999
    df = df.filter(F.col("customer_id") != -999999).na.drop(subset=["terminal_id"])
    
    # 3 - обработать выбросы, у нас только сверху
    Q1 = df.approxQuantile("tx_amount", [0.25], 0.05)[0]
    Q3 = df.approxQuantile("tx_amount", [0.75], 0.05)[0]
    IQR = Q3 - Q1
    ub = Q3 + 1.5 * IQR
    
    df = df.filter(F.col("tx_amount") <= ub)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument(
        "--file_path", 
        type=str, 
        help="file path of the data to be cleaned",
        required=True
        )
    
    args = parser.parse_args()
    main(args.file_path)
