from io import BytesIO
import PIL
from json import loads

from pyspark.sql import SparkSession
import pyspark.sql.types as T

from tsdb.ml.types import ImageMetadata


def get_image_metadata(image_binary: bytes) -> ImageMetadata:  # pragma: no cover
        # Try to read the image and if we fail, we have to default to
        # to the null image case
        image_binary = BytesIO(image_binary)

        try:
            image = PIL.Image.open(image_binary)
            exif = image._getexif()

        except FileNotFoundError:
            exif = None
        except UnicodeDecodeError:
            exif = None

        user_comment_exif_id = 37510

        if exif is None or user_comment_exif_id not in exif:
            # we need to return with default values
            fake_image = PIL.Image.new("RGB", (640, 640), "black")
            return {
                "height": 640,
                "width": 640,
                "lat": 0.0,
                "long": 0.0,
                "image_id": -1,
                "map_provider": "unknown",
                "image": fake_image
            }
        
        try:
            user_comment_exif = exif[user_comment_exif_id]
            exif_dict = loads(
                user_comment_exif.decode("utf-8").replace("\'", "\"")
            )
        
        except UnicodeDecodeError as e:
            # can we gracefully handle this?
            raise ValueError(f"Unable to decode exif data: {e}")
        
        image_id = -1 if "id" not in exif_dict else int(exif_dict["id"])
        return {
            "height": image.height,
            "width": image.width,
            "lat": exif_dict["lat"],
            "long": exif_dict["lng"],
            "image_id": image_id,
            "map_provider": exif_dict["mapProvider"],
            "image": image
        }
