# Databricks notebook source
##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook trains/logs a custom PyFunc model for Semantic Search.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``sample_project/resources/model-workflow-resource.yml``
#
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_data_path",
    "/databricks-datasets/faqs",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-sample_project-experiment",
    label="MLflow experiment name",
)

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev.sample_project.faq_semantic_search_model", label="Full (Three-Level) Model Name"
)

# FAQ features table name
dbutils.widgets.text(
    "faq_features_table",
    "dev.sample_project.faq_features",
    label="FAQ Features Table",
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
faq_features_table = dbutils.widgets.get("faq_features_table")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# DBTITLE 1, Define Custom PyFunc Model
import pandas as pd
import numpy as np

class FAQSearchModel(mlflow.pyfunc.PythonModel):
    def __init__(self, faq_features_table):
        self.faq_features_table = faq_features_table

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        # Load the sentence transformer model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def predict(self, context, model_input):
        '''
        model_input is expected to be a pandas DataFrame with a 'query' column.
        We return the top 5 most similar FAQs for each query.
        '''
        import pyspark
        from pyspark.sql import SparkSession
        # Get active spark session to query the feature table
        # In a real-world scenario, you might query a Vector Search Index instead of loading the Delta table
        spark = SparkSession.builder.getOrCreate()
        
        # Load FAQ data and embeddings
        faq_df = spark.table(self.faq_features_table).toPandas()
        
        results = []
        for query in model_input["query"]:
            # Encode query
            query_emb = self.encoder.encode(query)
            
            # Compute similarities
            similarities = []
            for idx, row in faq_df.iterrows():
                # Extract the embedding which is an array of floats
                faq_emb = np.array(row['embedding'])
                sim = self._cosine_similarity(query_emb, faq_emb)
                similarities.append((row['question'], row['answer'], sim))
                
            # Sort by similarity desc and get top 5
            similarities.sort(key=lambda x: x[2], reverse=True)
            top_5 = similarities[:5]
            
            # Format output
            formatted_res = [{"question": q, "answer": a, "score": float(s)} for q, a, s in top_5]
            results.append(formatted_res)
            
        return pd.Series(results)

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Start an mlflow run
mlflow.end_run()
mlflow.start_run()

from mlflow.models.signature import infer_signature

# Create a sample input to infer signature
sample_input = pd.DataFrame({"query": ["What is your return policy?"]})
sample_output = pd.Series([[{"question": "dummy", "answer": "dummy", "score": 1.0}]])
signature = infer_signature(sample_input, sample_output)

# Initialize and log model
faq_search_model = FAQSearchModel(faq_features_table=faq_features_table)

# In order to log, we can specify pip requirements or conda env
pip_requirements = [
    "sentence-transformers>=2.2.2",
    "pandas",
    "numpy",
    "pyspark"
]

mlflow.pyfunc.log_model(
    artifact_path="model_packaged",
    python_model=faq_search_model,
    registered_model_name=model_name,
    signature=signature,
    pip_requirements=pip_requirements
)

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
