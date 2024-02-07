#!/usr/bin/env python
import json
from typing import Dict, NamedTuple
import random
import logging
from datetime import datetime
import argparse
from collections import namedtuple

import kafka


class RecordMetadata(NamedTuple):
    topic: str
    partition: int
    offset: int


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
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
    argparser.add_argument(
        "-n",
        default=10,
        type=int,
        help="number of messages to send",
    )

    args = argparser.parse_args()

    producer = kafka.KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=args.user,
        sasl_plain_password=args.password,
        ssl_cafile="/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt",
        value_serializer=serialize,
    )

    try:
        for i in range(args.n):
            record_md = send_message(producer, args.topic)
            print(
                f"Msg sent. Topic: {record_md.topic}, partition:{record_md.partition}, offset:{record_md.offset}"
            )
    except kafka.errors.KafkaError as err:
        logging.exception(err)
    producer.flush()
    producer.close()


def send_message(producer: kafka.KafkaProducer, topic: str) -> RecordMetadata:
    txs = generate_txs()
    future = producer.send(
        topic=topic,
        value=txs,
    )

    # Block for 'synchronous' sends
    record_metadata = future.get(timeout=1)
    return RecordMetadata(
        topic=record_metadata.topic,
        partition=record_metadata.partition,
        offset=record_metadata.offset,
    )


def generate_txs(cnt=5) -> Dict:
    start = random.randint(0, 30)
    txs = [(i+1510422180, str(datetime.now()), random.randint(0, 909997), random.randint(0, 999), round(random.uniform(1, 7000), 2)) for i in range(start, cnt+start)]
    return txs


def serialize(msg: Dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


if __name__ == "__main__":
    main()
