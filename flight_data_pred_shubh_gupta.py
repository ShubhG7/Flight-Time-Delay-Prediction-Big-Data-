import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, hour, month, lit, to_timestamp, format_string, when, isnan
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from google.cloud import storage
from pyspark.sql.functions import count

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to save files to Google Cloud Storage
def save_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Complete Flight Delay Prediction with Visualizations and Error Handling") \
    .getOrCreate()
logging.info("Spark session started.")

# Define the schema based on your structure
schema = StructType([
    StructField('FL_DATE', TimestampType(), True),
    StructField('OP_CARRIER', StringType(), True),
    StructField('OP_CARRIER_FL_NUM', IntegerType(), True),
    StructField('ORIGIN', StringType(), True),
    StructField('DEST', StringType(), True),
    StructField('CRS_DEP_TIME', DoubleType(), True),
    StructField('DEP_TIME', DoubleType(), True),
    StructField('DEP_DELAY', DoubleType(), True),
    StructField('TAXI_OUT', DoubleType(), True),
    StructField('WHEELS_OFF', DoubleType(), True),
    StructField('WHEELS_ON', DoubleType(), True),
    StructField('TAXI_IN', DoubleType(), True),
    StructField('CRS_ARR_TIME', DoubleType(), True),
    StructField('ARR_TIME', DoubleType(), True),
    StructField('ARR_DELAY', DoubleType(), True),
    StructField('CANCELLED', DoubleType(), True),
    StructField('CANCELLATION_CODE', StringType(), True),
    StructField('DIVERTED', DoubleType(), True),
    StructField('CRS_ELAPSED_TIME', DoubleType(), True),
    StructField('ACTUAL_ELAPSED_TIME', DoubleType(), True),
    StructField('AIR_TIME', DoubleType(), True),
    StructField('DISTANCE', DoubleType(), True),
    StructField('CARRIER_DELAY', DoubleType(), True),
    StructField('WEATHER_DELAY', DoubleType(), True),
    StructField('NAS_DELAY', DoubleType(), True),
    StructField('SECURITY_DELAY', DoubleType(), True),
    StructField('LATE_AIRCRAFT_DELAY', DoubleType(), True),
    StructField('Unnamed: 27', StringType(), True)
])

try:
    # Load data
    flight_data = spark.read.csv("gs://shubhgassignmentsmetcs777/archive/2018.csv", header=True, inferSchema = True)
    logging.info("Data loaded successfully.")

    # Data preprocessing
    flight_data.select(*[(count(when((isnan(c) | col(c).isNull()), c)) if t not in ("timestamp", "date") else count(when(col(c).isNull(), c))).alias(c) for c, t in flight_data.dtypes if c in flight_data.columns ]).show()
    print(flight_data.schema)
    # flight_data = flight_data.dropna() 
    flight_data = flight_data.withColumn("CRS_DEP_TIME_STR", format_string("%04.0f", col("CRS_DEP_TIME")))
    flight_data.show(n=1)
    flight_data = flight_data.withColumn("CRS_DEP_TIME_TS", to_timestamp("CRS_DEP_TIME_STR", "HHmm"))
    flight_data.show(n=1)
    flight_data = flight_data.withColumn("day_of_week", dayofweek("FL_DATE"))
    flight_data.show(n=1)
    flight_data = flight_data.withColumn("hour_of_day", hour("CRS_DEP_TIME_TS"))
    flight_data.show(n=1)
    flight_data = flight_data.withColumn("month", month("FL_DATE"))
    flight_data.show(n=1)
    flight_data = flight_data.na.fill({"DEP_DELAY": 0})
    logging.info("Data preprocessing completed.")
    flight_data.show(n=5)
    print(flight_data.schema)

    # Feature engineering
    indexer = StringIndexer(inputCols=["OP_CARRIER", "ORIGIN", "DEST"], outputCols=["carrier_index", "origin_index", "dest_index"], handleInvalid="skip")
    flight_data = indexer.fit(flight_data).transform(flight_data)
    flight_data.show(n=5)
    
    assembler = VectorAssembler(
        inputCols=["day_of_week", "hour_of_day", "month", "carrier_index", "origin_index", "dest_index", "DISTANCE"],
        outputCol="features",
        handleInvalid="skip"  # Handle nulls by skipping them
    )
    feature_vector = assembler.transform(flight_data)
    logging.info("Features assembled.")

    # Split data
    train_data, test_data = feature_vector.randomSplit([0.8, 0.2], seed=42)
    
    print(train_data.count())

    # Model setup and hyperparameter tuning
    rf = RandomForestRegressor(featuresCol="features", labelCol="DEP_DELAY")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [100]) \
        .addGrid(rf.maxDepth, [10]) \
        .addGrid(rf.maxBins, [400]) \
        .build()
    tvs = TrainValidationSplit(estimator=rf,
                            estimatorParamMaps=paramGrid,
                            evaluator=RegressionEvaluator(labelCol="DEP_DELAY", predictionCol="prediction", metricName="rmse"),
                            trainRatio=0.8)
    # tvs.setParallelism(2)
    # logging.info("Parallelism Achieved")
    best_model = tvs.fit(train_data)
    logging.info("Model trained with best parameters.")

    # Model evaluation
    predictions = best_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="DEP_DELAY", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    results_df = predictions.withColumn("RMSE", lit(rmse))
    # results_df.select("DEP_DELAY", "prediction", "RMSE").write.format("csv").option("header", True).save("gs://shubhgassignmentsmetcs777/airline_delay_prediction/predictions_and_metrics.csv")
    logging.info("Predictions and RMSE saved as CSV to GCS.")
    
    # Convert predictions to pandas DataFrame for plotting
    pd_df = predictions.select("DEP_DELAY", "prediction").toPandas()
    pd_df_complete = predictions.select("day_of_week", "hour_of_day", "month", "carrier_index", "origin_index", "dest_index", "DISTANCE", "DEP_DELAY", "prediction").toPandas()

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=best_model.bestModel.featureImportances.toArray(), y=["day_of_week", "hour_of_day", "month", "carrier_index", "origin_index", "dest_index", "DISTANCE"])
    plt.title('Feature Importances')
    plt.savefig("/tmp/feature_importances.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(pd_df['DEP_DELAY'], pd_df['prediction'], alpha=0.5)
    plt.xlabel('Actual Delays')
    plt.ylabel('Predicted Delays')
    plt.title('Actual vs Predicted Delays')
    plt.savefig("/tmp/actual_vs_predicted.png")
    plt.close()

    corr_matrix = pd_df_complete.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig("/tmp/correlation_heatmap.png")
    plt.close()

    save_to_gcs('shubhgassignmentsmetcs777', '/tmp/feature_importances.png', 'airline_delay_prediction/visualizations/feature_importances.png')
    save_to_gcs('shubhgassignmentsmetcs777', '/tmp/actual_vs_predicted.png', 'airline_delay_prediction/visualizations/actual_vs_predicted.png')
    save_to_gcs('shubhgassignmentsmetcs777', '/tmp/correlation_heatmap.png', 'airline_delay_prediction/visualizations/correlation_heatmap.png')
    logging.info("Visualizations saved to GCS.")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")

# Stop Spark session
spark.stop()
logging.info("Spark session stopped.")
