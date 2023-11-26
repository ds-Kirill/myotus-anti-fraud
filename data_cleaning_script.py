from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

def main(file_path):
    # Инициализация SparkSession
    spark = SparkSession.builder.appName("AntiFraudOTUS").getOrCreate()

    # Загрузка данных
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
    .csv(file_path)

    # Вызов функции для очистки данных
    cleaned_df = clean_data(df)

    # Сохранение очищенных данных
    # Пример: cleaned_df.write.csv("path_to_save_cleaned_data.csv", header=True)
    new_path = "clean_data.parquet"
    cleaned_df.write.parquet(new_path) # ваш код для сохранения данных

    # Закрытие SparkSession
    spark.stop()

def clean_data(df):
    # Функции для очистки данных
    
    # 1 -убрать дубли
    df = df.dropDuplicates() 
    
    # 2 - убрать пропуски, в том числе customer_id = -9999999
    df = df.filter(F.col("customer_id") != -999999).na.drop(subset["terminal_id"])
    
    # 3 - обработать выбросы, у нас только сверху
    Q1 = df.approxQuantile("tx_amount", [0.25], 0.05)[0]
    Q3 = df.approxQuantile("tx_amount", [0.75], 0.05)[0]
    IQR = Q3 - Q1
    ub = Q3 + 1.5 * IQR
    
    df_clean = df.filter(F.col("tx_amount") <= ub)
    
    return df_clean


if __name__ == "__main__":
    main()