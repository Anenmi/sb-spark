# lab03 solution (anenmi)
# coding: utf-8


import os
import sys
import json 

os.environ["PYSPARK_PYTHON"]='/opt/anaconda/envs/bd9/bin/python'
os.environ["SPARK_HOME"]='/usr/hdp/current/spark2-client'

sparkClassPath = os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.postgresql:postgresql:42.2.12,com.datastax.spark:spark-cassandra-connector_2.11:2.5.1,org.apache.spark:spark-sql_2.11:2.4.5,org.apache.spark:spark-core_2.11:2.4.5,org.elasticsearch:elasticsearch-spark-20_2.11:6.8.9    --conf spark.sql.extensions=com.datastax.spark.connector.CassandraSparkExtensions     --conf spark.cassandra.connection.host=10.0.0.31:9042     --num-executors 2 pyspark-shell'
spark_home = os.environ.get('SPARK_HOME', None)

if not spark_home:
    raise ValueError('SPARK_HOME environment variable is not set')

sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.10.7-src.zip'))
exec(open(os.path.join(spark_home, 'python/pyspark/shell.py')).read())

from pyspark import SparkContext, SQLContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark import Row
import json

conf = SparkConf()

username = 'an_en'
password = 'QjFvQaAt'

spark = SparkSession \
    .builder \
    .appName("postgresql") \
    .config("spark.driver.extraClassPath", sparkClassPath) \
    .getOrCreate()
print('SparkSession started')

print('load clients table...')
cl = spark.read\
    .format("org.apache.spark.sql.cassandra")\
    .options(table='clients', keyspace='labdata')\
    .load()

cl_ = cl.select("uid", 
                "gender", 
                F.expr("CASE WHEN age <= 24 THEN '18-24' \
                        WHEN age <= 34 THEN '25-34' \
                        WHEN age <= 44 THEN '35-44' \
                        WHEN age <= 54 THEN '45-54' \
                        ELSE '>=55' END AS age_cat"))

print('load domain_cats table...')
cats = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://10.0.0.31:5432/labdata") \
    .option("dbtable", "domain_cats") \
    .option("user", username) \
    .option("password", password) \
    .option("driver", "org.postgresql.Driver") \
    .load()

cats_ = cats.withColumn('category', F.concat(F.lit('web_'), F.col('category')))

print('load visits table...')
visits = spark.read\
    .format("org.elasticsearch.spark.sql")\
    .option("es.nodes","10.0.0.31")\
    .option("es.port","9200")\
    .option("user", username) \
    .option("password", password) \
    .load("visits")

visits_ = visits\
    .where("uid is not null")\
    .groupBy(["uid", "category"])\
    .count()\
    .withColumn('category', F.concat(F.lit('shop_'), 
                                     F.regexp_replace(F.regexp_replace(F.lower(F.col('category')), 
                                                                       '-', '_'), 
                                                      ' ', '_')))\

visits_pivot = visits_\
    .groupBy("uid")\
    .pivot("category")\
    .agg(F.sum("count"))\
    .na.fill(value=0)

print('load weblogs table...')
web = sc.textFile("hdfs:///labs/laba03/weblogs.json")
web = web.map(lambda x: json.loads(x))

schema = StructType([
    StructField('uid', StringType(), True),
    StructField('visits', ArrayType(StringType()), True)
])

web_df = spark.createDataFrame(web, schema=schema)

my_udf = F.udf(lambda webs: [(web.split('/')[2].replace("www.", "")) for web in webs], 
               ArrayType(StringType()))

web_df_ = web_df\
    .withColumn("domains", my_udf(F.col('visits')))\
    .withColumn("domain", F.explode(F.col('domains')))\
    .select('uid', 'domain')

web_df_ = web_df_.join(cats_, cats_.domain == web_df_.domain, 'inner')\
            .groupBy(["uid", "category"])\
            .count()

web_pivot = web_df_\
    .groupBy("uid")\
    .pivot("category")\
    .agg(F.sum("count"))\
    .na.fill(value=0)

res = cl_\
    .join(visits_pivot, cl.uid == visits_pivot.uid, 'left')\
    .join(web_pivot, cl.uid == web_pivot.uid, 'left')\
    .na.fill(value=0)\
    .drop(visits_pivot.uid)\
    .drop(web_pivot.uid)

print('starting to compute...')
res.write    .format("jdbc")    .option("url", "jdbc:postgresql://10.0.0.31:5432/an_en")    .option("dbtable", "clients")    .option("user", username)    .option("password", password)    .option("driver", "org.postgresql.Driver")    .save()
print('result was overwrite to postgresql')


