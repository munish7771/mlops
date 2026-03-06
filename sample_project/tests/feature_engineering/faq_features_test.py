import pyspark.sql
import pytest
from pyspark.sql import SparkSession

from sample_project.feature_engineering.features.faq_features import compute_features_fn


@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session"""
    spark = (
        SparkSession.builder.appName("pytest-pyspark-local-testing")
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
