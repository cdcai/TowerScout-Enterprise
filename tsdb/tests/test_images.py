import pytest
from pyspark.sql import SparkSession
from PIL import Image

from tsdb.preprocessing.images import get_image_metadata


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_data_processing")
        .getOrCreate()
    )
    
    return spark


@pytest.fixture
def image_binary_dir(spark: SparkSession) -> str:
    if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
        congfigs = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
        catalog = congfigs["catalog_name"]

    else:
        RaiseException("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

    return f"/Volumes/{catalog}/towerscout/misc/unit_tests/image_binary_dataset/"



def test_get_image_metadata(spark: SparkSession, image_binary_dir: str):
    image_df = (
    spark
    .read
    .format("binaryFile")
    .load(image_binary_dir)
    .select("content")
    .limit(1)
    )

    image_df = image_df.toPandas()
    image_bin = image_df["content"][0]
    image_metadata = get_image_metadata(image_bin)

    assert isinstance(image_metadata, dict), "Image metadata is not a dictionary"

    keys_to_check = ("width", "height", "image", "lat", "long", "image_id", "map_provider", "image")
    assert all(key in image_metadata for key in keys_to_check), f"Missing key(s) from {keys_to_check} in {image_metadata}" 
    
    assert isinstance(image_metadata["image"], Image.Image), "Image value is not a PIL.Image object"

