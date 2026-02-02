import pytest
from pyspark.sql import SparkSession

from tsdb.ml.validate import update_benchmark_table


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_validate_functions")
        .getOrCreate()
    )

    return spark


@pytest.fixture()
def db_args(spark: SparkSession) -> list[str]:
    if spark.catalog._jcatalog.tableExists(
        "global_temp.global_temp_towerscout_configs"
    ):
        configs = spark.sql(
            "SELECT * FROM global_temp.global_temp_towerscout_configs"
        ).collect()[0]
        catalog = configs["catalog_name"]
        schema = configs["schema_name"]
        return [catalog, schema, "benchmark_results"]
    else:
        raise Exception(
            "Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook"
        )


@pytest.fixture()
def benchmark_data(db_args):
    overall_metrics = {'f1': 0.796, 'precision': 0.906, 'recall': 0.710}
    per_class_metrics = {'ct': {'f1': 0.7967, 'precision': 0.906, 'recall': 0.710}}
    model_metadata = {"uc_model_name": "Model A", "uc_model_version": 1, "model_uri": "s3://models/14244/model_a"}

    return f"{db_args[0]}.{db_args[1]}.{db_args[2]}", overall_metrics, per_class_metrics, model_metadata


def test_update_benchmark_table(spark: SparkSession, benchmark_data) -> None:
    """Test that the benchmark table is updated correctly"""
    update_benchmark_table(*benchmark_data)
    
    df = spark.sql(f"SELECT * FROM {benchmark_data[0]} WHERE model_metadata.model_uri = 's3://models/14244/model_a'")
    
    assert (
        df.count() == 1
    ), f"Exactly 1 record should have been added to the benchmark table but actual number added was {df.count()}."

    # clean up
    spark.sql(f"DELETE FROM {benchmark_data[0]} WHERE model_metadata.model_uri = 's3://models/14244/model_a'")