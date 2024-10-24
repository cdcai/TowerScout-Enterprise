import pytest
import json

from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_functions_transformation")
        .getOrCreate()
    )

    return spark


def test_merge_into_gold_query(spark: SparkSession) -> None:
    """
    Tests the cast_to_column function with a string input and verifies it returns
    a column object. Column objects are oftenmore useful than strings but we still
    want the flexibility of being able to pass a column name as an input.
    """
    validated_data = (
        (
            "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k121.jpg",
            "ab3c5de",
            {0: {'label':0, 'xmin':8.2, 'xmax':5.4, 'ymin':5.1, 'ymax':9.2}, 1: {'labe':0, 'xmin':0.2, 'xmax':3.3, 'ymin':55.4, 'ymax':3.5}}
        ),
        (
            "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k119.jpg",
            "g4mf8n",
            {0: {'label':0, 'xmin':1.2, 'xmax':75.4, 'ymin':55.1, 'ymax':98.2}, 1: {'labe':0, 'xmin':1.2, 'xmax':2.3, 'ymin':3.4, 'ymax':7.5}}
        )
    )

    values = ", ".join(
        [f"('{path}', '{img_hash}', '{json.dumps(bboxes)}')" for (path, img_hash, bboxes) in validated_data]
    )

    paths = ", ".join([f"'{path}'" for (path, img_hash, bboxes) in validated_data])
    delete_existing = "DELETE FROM edav_dev_csels.towerscout_test_schema.test_image_gold WHERE path IN ({paths});"
    spark.sql(delete_existing) # delete exisiting record from test gold table for test

    drop_existing_view = "DROP VIEW IF EXISTS gold_updates;"
    spark.sql(delete_existing)

    create_updates_view = f"""
            CREATE TEMPORARY VIEW gold_updates AS
            WITH temp_data(paths, imgHash, bboxs) AS (
            VALUES
                {values}
            )

            SELECT from_json(temp.bboxs, 'annotations array<struct<`xmin`:float, `xmax`:float, `ymim`:float, `ymax`:float, `label`:int>>') as bbox, temp.uuid, temp.imgHash, silver.length
            FROM edav_dev_csels.towerscout_test_schema.test_image_silver AS silver
            JOIN temp_data AS temp
            ON silver.path = temp.path
            WHERE silver.path in ({paths});
            """
    spark.sql(create_updates_view)
    
    
    assert isinstance(
        result, F.Column
    ), f"Expected a PySpark Column object, got {type(result)}"