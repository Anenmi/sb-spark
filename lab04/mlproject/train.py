
# coding: utf-8
# Обучение

import os
import sys
import json 

os.environ["PYSPARK_PYTHON"]='/opt/anaconda/envs/bd9/bin/python'
os.environ["SPARK_HOME"]='/usr/hdp/current/spark2-client'

sparkClassPath = os.environ['PYSPARK_SUBMIT_ARGS'] = '--num-executors 2 pyspark-shell'
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
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import CountVectorizer, StringIndexer, IndexToString
from pyspark.ml import Pipeline, util


conf = SparkConf().setAll([('spark.train_df_path', '/labs/laba_ds04/laba04.json'), 
                          ('spark.model_path', '/user/an.en/sb-spark/lab04/mlproject')])

spark = (SparkSession
         .builder
         .config(conf=conf)
         .appName("fit")
         .getOrCreate())

### 1 export data
row_data = sc.textFile(conf.get("spark.train_df_path"))
row_data_map = row_data.map(lambda x: json.loads(x))
schema = StructType([
    StructField('uid', StringType(), True),
    StructField('gender_age', StringType(), True),
    StructField('visits', ArrayType(StringType()), True)
])
train = spark.createDataFrame(row_data_map, schema=schema)

### 2 transform data
my_udf = F.udf(lambda webs: [(web.split('/')[2].replace("www.", "")) for web in webs], 
               ArrayType(StringType()))
train = train\
    .withColumn("domains", my_udf(F.col('visits')))\
    .select('uid', 'domains', 'gender_age')


### 3 build pipeline
cv = CountVectorizer(inputCol="domains", 
                     outputCol="features")
indexer = StringIndexer(inputCol="gender_age", 
                        outputCol="label")
lr = LogisticRegression(maxIter=10, 
                        regParam=0.001)
pipeline = Pipeline(stages=[cv, 
                            indexer, 
                            lr
                           ])
### 4 train model
model = pipeline.fit(train)

model.write().overwrite().save(conf.get("spark.model_path"))

spark.stop()