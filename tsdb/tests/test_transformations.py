import pytest
import tsdb.preprocessing.transformations as processing

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, StructType, StructField
import pyspark.sql.functions as F


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


@pytest.fixture(scope="module")
def image_df(spark) -> DataFrame:
    data = [
        ("img0", "bbox0"),
        ("img1", "bbox1"),
        ("img2", "bbox2"),
        ("img3", "bbox3"),
        ("img4", "bbox4"),
        ("img5", "bbox5"),
        ("img6", "bbox6"),
        ("img7", "bbox7"),
        ("img8", "bbox8"),
        ("img9", "bbox9"),
    ]

    return spark.createDataFrame(data, ["image", "bbox"])


def test_split_ratios(spark, image_df):
    """
    Test the split ratio. Random Split seems to work approximately, so
    we use approx to check if the ratios are approximately equal. 
    """
    # Force everything onto 1 partitation, may potentially resolve issue with wrong splits
    image_df = image_df.repartition(1)
    train_df, test_df, val_df = processing.train_test_val_split(image_df, 0.6, 0.3, 0.1)

    # Check if the splits respect the ratios approximately
    total_count = image_df.count()
    train_count = train_df.count()
    test_count = test_df.count()
    val_count = val_df.count()

    assert total_count == train_count + test_count + val_count

    absolute_tolerance = 0.25
    assert pytest.approx(train_count / total_count, abs=absolute_tolerance) == 0.6
    assert pytest.approx(test_count / total_count, abs=absolute_tolerance) == 0.3
    assert pytest.approx(val_count / total_count, abs=absolute_tolerance) == 0.1
    

def test_bounding_boxes_intact(spark, image_df):
    """
    Test if bounding boxes remain intact with images aka no overlaps between splits
    """
    train_df, test_df, val_df = processing.train_test_val_split(image_df, 0.6, 0.3, 0.1)

    # Check if the bounding boxes associated with an image stay in the same split
    def collect_bboxes(df):
        return [
            row[0] for row in df.select("bbox").distinct().collect()
        ]
    
    train_bboxes = collect_bboxes(train_df)
    test_bboxes = collect_bboxes(test_df)
    val_bboxes = collect_bboxes(val_df)

    assert not set(train_bboxes) & set(test_bboxes) & set(val_bboxes)


def test_empty_df(spark):
    """
    Test that an empty DataFrame returns empty splits
    """
    schema = StructType([
        StructField("image", StringType(), True),
        StructField("bbox", StringType(), True)
    ])
    empty_df = spark.createDataFrame([], schema)
    train_df, test_df, val_df = processing.train_test_val_split(empty_df, 0.6, 0.3, 0.1)

    assert train_df.count() == 0
    assert test_df.count() == 0
    assert val_df.count() == 0


def test_single_image(spark):
    """
    Test that a dataframe with a single row returns that element in the train set
    and empty splits for the remaining.
    """
    # This test sometimes fails because the train_df is empty, hard to reproduce
    single_image_df = spark.createDataFrame([("img0", "bbox0")], ["str", "bbox"])
    train_df, test_df, val_df = processing.train_test_val_split(single_image_df, 0.6, 0.3, 0.1)
        
    # Since there's only 1 image, all of it should go into train
    assert train_df.count() == 1
    assert test_df.count() == 0
    assert val_df.count() == 0
