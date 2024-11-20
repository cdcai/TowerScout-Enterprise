from io import BytesIO
from PIL import Image
from json import loads

from pyspark.sql import SparkSession
import pyspark.sql.types as T


def make_image_metadata_udf(spark: SparkSession):
    towerscout_image_metadata_schema = T.StructType([
        T.StructField("height", T.IntegerType()),
        T.StructField("width", T.IntegerType()),
        T.StructField("lat", T.DoubleType()),
        T.StructField("long", T.DoubleType()),
        T.StructField("image_id", T.IntegerType()),
        T.StructField("map_provider", T.StringType())
    ])

    def get_image_metadata(image_binary: T.BinaryType):
        image = Image.open(BytesIO(image_binary))
        exif = image._getexif()
        user_comment_exif_id = 37510

        if exif is None or user_comment_exif_id not in exif:
            # we need to return with default values
            return {
                "height": image.height,
                "width": image.width,
                "lat": 0.0,
                "long": 0.0,
                "id": -1,
                "map_provider": "unknown"
            }
        
        try:
            user_comment_exif = exif[user_comment_exif_id]
            exif_dict = loads(
                user_comment_exif.decode("utf-8").replace("\'", "\"")
            )
        
        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to decode exif data: {e}")
        
        image_id = -1 if "id" not in exif_dict else int(exif_dict["id"])
        return {
            "height": image.height,
            "width": image.width,
            "lat": exif_dict["lat"],
            "long": exif_dict["lng"],
            "image_id": image_id,
            "map_provider": exif_dict["mapProvider"]
        }
    
    return spark.udf.register(
        "get_image_metadata_udf",
        get_image_metadata, 
        towerscout_image_metadata_schema
    )
