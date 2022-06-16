
# coding: utf-8

import os
import sys
import json 

os.environ["PYSPARK_PYTHON"]='/opt/anaconda/envs/bd9/bin/python'
os.environ["SPARK_HOME"]='/usr/hdp/current/spark2-client'

sparkClassPath = os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.0,\
org.apache.kafka:kafka-clients:2.8.1 \
--num-executors 2 pyspark-shell'
spark_home = os.environ.get('SPARK_HOME', None)

if not spark_home:
    raise ValueError('SPARK_HOME environment variable is not set')

sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.10.7-src.zip'))
exec(open(os.path.join(spark_home, 'python/pyspark/shell.py')).read())


from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import CountVectorizer, StringIndexer, IndexToString
from pyspark.ml import Pipeline, util, PipelineModel
import re 

conf = SparkConf().setAll([('spark.model_path', '/user/an.en/sb-spark/lab04/mlproject'),
                          ('spark.input_topic', 'an_en'),
                          ('spark.output_topic', 'an_en_lab04_out')])

spark = (SparkSession
         .builder
         .config(conf=conf)
         .config("spark.driver.extraClassPath", sparkClassPath) 
         .appName("predict")
         .getOrCreate())

uid_udf = F.udf(lambda x: re.match('"uid": "(.*)",', x), StringType())
web_udf = F.udf(lambda webs: [(web.split('/')[2].replace("www.", "")) for web in webs], 
               ArrayType(StringType()))

model = PipelineModel.load(conf.get("spark.model_path"))
stringer = IndexToString(inputCol="prediction", 
                        outputCol="gender_age"
                        )
stringer.setLabels(model.stages[1].labels)

schema = StructType([
    StructField("uid", StringType()),
    StructField("visits", ArrayType(StringType()))
])

test = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "10.0.0.31:6667") \
    .option("subscribe", conf.get("spark.input_topic")) \
    .option("includeHeaders", "true") \
    .load()\
    .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")\
    .withColumn("value", F.from_json(F.col("value").cast("string"), schema))\
    .withColumn("domains", web_udf(F.col('value.visits')))\
    .withColumn("uid", F.col('value.uid'))\
    .select('uid', 'domains')

result = model.transform(test)
result = stringer.transform(result)

query = result\
    .select(F.concat(F.lit('{"uid": "'), F.col('uid'),
                     F.lit('", "gender_age": "'), F.col('gender_age'), F.lit('"}')).alias('value'))\
    .selectExpr("CAST(value AS STRING)") \
    .writeStream\
    .format("kafka")\
    .option("kafka.bootstrap.servers", "10.0.0.31:6667") \
    .option("topic", conf.get("spark.output_topic")) \
    .option("checkpointLocation", '/user/an.en/sb-spark/lab04/mlproject/checkpoint')\
    .start()


print("start")
query.awaitTermination(10000)

spark.stop()