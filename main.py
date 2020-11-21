#imports
import pandas as pd
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, VectorIndexer,StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# main source used for this code : https://docs.databricks.com/applications/machine-learning/preprocess-data/transfer-learning-tensorflow.html


#mounting ADLSGen2
configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": "*****", # this comes from your application
           "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope = "***", key = "***"),
           "fs.azure.account.oauth2.client.endpoint": "****"} # this comes from your subscription


# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "****",
  mount_point = "/mnt/de",
  extra_configs = configs)


# read in the files from the mounted storage as binary file
images_df = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load('dbfs:/mnt/de/path_to_images')


# select the base model, here I have used ResNet50
model = ResNet50(include_top=False)
model.summary()  # verify that the top layer is removed

bc_model_weights = sc.broadcast(model.get_weights())

#declaring functions to execute on the worker nodes of the Spark cluster
def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model = ResNet50(weights=None, include_top=False)
  model.set_weights(bc_model_weights.value)
  return model

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  output = [p.flatten() for p in preds]
  return pd.Series(output)


@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)


# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")


# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
features_df = images_df.repartition(16).select(col("path"), featurize_udf("content").alias("features"))


#MLLib needs some post processing of the features column format
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
features_df = features_df.select(
   col("path"),  
    list_to_vector_udf(features_df["features"]).alias("features")
)

# OMITTED HERE
# You need to add the labels to your dataset based on the path of your images

# splitting in to training, validate and test set
df_train_split, df_validate_split, df_test_split =  features_df.randomSplit([0.6, 0.3, 0.1],42)  


#Here we start to train the tail of the model

# This concatenates all feature columns into a single feature vector in a new column "featuresModel".
vectorAssembler = VectorAssembler(inputCols=['features'], outputCol="featuresModel")

labelIndexer = StringIndexer(inputCol="Target", outputCol="indexedTarget").fit(features_df)

lr = LogisticRegression(maxIter=5, regParam=0.03, 
                        elasticNetParam=0.5, labelCol="indexedTarget", featuresCol="featuresModel")

# define a pipeline model
sparkdn = Pipeline(stages=[labelIndexer,vectorAssembler,lr])
spark_model = sparkdn.fit(df_train_split) # start fitting or training

# evaluating the model
predictions = spark_model.transform(df_test_split)

# Select example rows to display.
predictions.select("prediction", "indexedTarget", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedTarget", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
