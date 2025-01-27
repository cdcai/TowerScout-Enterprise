import pytest
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StructType, StructField, BinaryType
from tsdb.preprocessing.functions import image_statistics_udf, compute_image_statistics
from PIL import Image
import io
from tsdb.preprocessing.functions import statistics_schema

@pytest.fixture(scope="module")
def spark_session() -> SparkSession:
    """
    Provides a SparkSession for testing.
    """
    spark: SparkSession = (
        SparkSession.builder.master("local")
        .appName("test_preprocessing_functions")
        .getOrCreate()
    )
    return spark

@pytest.fixture(scope="module")
def image_data() -> bytes:
    """
    Creates a small image for testing.
    """
    img: Image.Image = Image.new("RGB", (10, 10), color="blue")
    img_bytes: io.BytesIO = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

def test_image_statistics_udf(image_data: bytes) -> None:
    """
    Test the `image_statistics_udf` function with a valid image binary.
    """
    result: dict = image_statistics_udf(image_data)
    assert isinstance(result, dict), "Expected the result to be a dictionary."
    assert "mean" in result, "The result should contain 'mean'."
    assert "median" in result, "The result should contain 'median'."
    assert "stddev" in result, "The result should contain 'stddev'."
    assert "extrema" in result, "The result should contain 'extrema'."

    assert len(result["mean"]) == 3, "Expected RGB mean values."
    assert len(result["median"]) == 3, "Expected RGB median values."
    assert len(result["stddev"]) == 3, "Expected RGB stddev values."
    assert len(result["extrema"]) == 3, "Expected RGB extrema values."

@pytest.fixture(scope="module")
def image_dataframe(spark_session: SparkSession, image_data: bytes) -> DataFrame:
    """
    Creates a DataFrame with binary image data for testing.
    """
    data: list[tuple[bytes]] = [(image_data,)]
    schema: StructType = StructType([StructField("image_column", BinaryType(), True)])
    return spark_session.createDataFrame(data, schema)


def test_compute_image_statistics(spark_session: SparkSession, image_dataframe: DataFrame) -> None:
    """
    Test the compute_image_statistics function.
    """
    spark_session.udf.register("image_statistics_udf", image_statistics_udf, statistics_schema)

    result_df: DataFrame = compute_image_statistics(image_dataframe, "image_column")
    result: list[Row] = result_df.select("statistics").collect()

    assert len(result) == 1, "Expected one row in the result."

    statistics: Row = result[0]["statistics"]

    assert isinstance(statistics, Row), f"Expected a Row, got {type(statistics)}"
    assert hasattr(statistics, "mean"), "Statistics should include 'mean'."
    assert hasattr(statistics, "median"), "Statistics should include 'median'."
    assert hasattr(statistics, "stddev"), "Statistics should include 'stddev'."
    assert hasattr(statistics, "extrema"), "Statistics should include 'extrema'."

    assert len(statistics.mean) == 3, "Expected mean to have 3 values for RGB."
    assert len(statistics.median) == 3, "Expected median to have 3 values for RGB."
    assert len(statistics.stddev) == 3, "Expected stddev to have 3 values for RGB."
    assert len(statistics.extrema) == 3, "Expected extrema to have 3 tuples for RGB."
