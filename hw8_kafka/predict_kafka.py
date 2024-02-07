#!/usr/bin/env python
import findspark
findspark.init()

import json
import argparse
from typing import Dict, NamedTuple
from kafka import KafkaConsumer
import kafka

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F

# from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType



def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
       "-g", "--group_id", required=True, help="kafka consumer group_id"
    )
    argparser.add_argument(
        "-b",
        "--bootstrap_server",
        default="rc1a-nra2ph2g1088r83i.mdb.yandexcloud.net:9091",
        help="kafka server address:port",
    )
    argparser.add_argument(
        "-u", "--user", default="fraud", help="kafka user"
    )
    argparser.add_argument(
        "-p", "--password", default="12345678!", help="kafka user password"
    )
    argparser.add_argument(
        "-t", "--topic", default="data", help="kafka topic to consume"
    )

    args = argparser.parse_args()

    consumer = KafkaConsumer(
        bootstrap_servers=args.bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=args.user,
        sasl_plain_password=args.password,
        ssl_cafile="/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt",
        group_id=args.group_id,
        value_deserializer=json.loads,
    )
    
    spark = SparkSession.builder.appName("SparkMLPredict").getOrCreate()    
    model = PipelineModel.load("s3a://fraudstop/artifacts/tree_md7/sparkml")
    struct = StructType([
            StructField("transaction_id", IntegerType(), nullable=True),
            StructField("tx_datetime", StringType(), nullable=True),
            StructField("customer_id", IntegerType(), nullable=True),
            StructField("terminal_id", IntegerType(), nullable=True),
            StructField("tx_amount", DoubleType(), nullable=True)
            ])  
            
    consumer.subscribe(topics=[args.topic])

    predict(consumer, spark, model, struct, bootstrap_server=args.bootstrap_server, user=args.user, password=args.password)
    
    spark.stop()

def predict(consumer, spark, model, struct, bootstrap_server, user, password):
    print("Waiting for a new messages. Press Ctrl+C to stop")
    
    for msg in consumer:
        print(msg.value)
        try:
            df = spark.createDataFrame(msg.value, struct)

            df = df.withColumn("tx_datetime", col("tx_datetime").cast("timestamp"))
            df = df.withColumn("tx_time",
                   F.unix_timestamp(col("tx_datetime")) - F.unix_timestamp(F.to_date(col("tx_datetime"))))
            df = df.select("transaction_id", "tx_time", "customer_id", "terminal_id", "tx_amount")
            print("make predictioin")
            predictions = model.transform(df)
            print("OK")
#            print(predictions.select("prediction"))
            prediction_rows = predictions.select("transaction_id", "prediction").collect()
            data_dict = dict(prediction_rows)
            write_topic(data_dict, bootstrap_server=bootstrap_server, user=user, password=password)

        except KeyboardInterrupt:
            pass

def write_topic(preds, bootstrap_server, user, password):
    producer = kafka.KafkaProducer(
        bootstrap_servers=bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=user,
        sasl_plain_password=password,
        ssl_cafile="/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt",
        value_serializer=serialize,
    )
    try:
        future = producer.send(
        topic="predicted",
        value=preds,
        )
        record_metadata = future.get(timeout=1)
        print(
                f"Msg sent. Topic: {record_metadata.topic}, partition:{record_metadata.partition}, offset:{record_metadata.offset}"
            )
        
    except kafka.errors.KafkaError as err:
        logging.exception(err)    
            
    producer.flush()
    producer.close()

def serialize(msg: Dict) -> bytes:
    return json.dumps(msg).encode("utf-8")

if __name__ == "__main__":
