import pyspark.sql
import pytest
import os
import sys
from pyspark.sql import SparkSession

# Ensure JAVA_HOME is set for PySpark in Windows Conda environments
if os.name == "nt":
    if not os.environ.get("JAVA_HOME"):
        os.environ["JAVA_HOME"] = os.path.join(sys.prefix, "Library")
    # Explicitly set PySpark config to prevent worker timeouts and DNS issues
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from sample_project.feature_engineering.features.faq_features import compute_features_fn


@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session"""
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-pyspark-local-testing")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())
    return spark


@pytest.mark.usefixtures("spark")
def test_compute_features_fn(spark):
    output_df = compute_features_fn(
        input_df=None, 
        timestamp_column="created_at", 
        start_date=None, 
        end_date=None
    )
    assert isinstance(output_df, pyspark.sql.DataFrame)
    assert output_df.count() > 0
    assert "embedding" in output_df.columns
    assert "created_at" in output_df.columns
