#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
# TowerScout Team:
# Karen Wong, Gunnar Mein, Thaddeus Segura, Jia Lu
#
# Licensed under CC-BY-NC-SA-4.0
# (see LICENSE.TXT in the root of the repository for details)
#

#
# the provider-independent part of maps
#
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import aiofiles
import requests
import mapbox_vector_tile
import json
import math
from io import BytesIO
from PIL import Image, ExifTags
from PIL import PngImagePlugin
import piexif
import io
import base64
import asyncio
import aiohttp
import aiofiles
import ssl
from collections import deque
import time
import uuid
from datetime import datetime
from azure.identity import DefaultAzureCredential
import logging, ts_secrets
from flask import request
import getpass

current_directory = os.getcwd()
config_dir = os.path.join(os.getcwd().replace("webapp", ""), "webapp")
timeout = aiohttp.ClientTimeout(total=600)  # Set the timeout to 6 minutes

class Map:

    def __init__(self):
        self.has_metadata = False

    def get_sat_maps(self, tiles, loop, fname, user_id):
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            if user_id == 'None':
                user_id = getpass.getuser()  # Get the username for local
            self.user_id = user_id
            urls = []
            for tile in tiles:
                # ask provider for this specific url
                url = self.get_url(tile)
                urls.append(url)
                tile["url"] = url
                # print(urls[-1])
                if self.has_metadata:
                    urls.append(self.get_meta_url(tile))
                    tile["metaurl"] = self.get_meta_url(tile)

            # tilesMetaData = self.getTilesMetaData(tiles=tiles)
            # execute
            unique_directory = generate_unique_directory_name(self)
            loop.run_until_complete(
                gather_urls(urls, fname, self.has_metadata, self.mapType, tiles, unique_directory, self)
            )
            return self.has_metadata, self.request_id
            
        except Exception as e:
            logging.error("Error at %s", "get_sat_maps ts_maps.py", exc_info=e)
        except RuntimeError as e:
            logging.error("Error at %s", "get_sat_maps ts_maps.py", exc_info=e) 
        except SyntaxError as e:
            logging.error("Error at %s", "get_sat_maps ts_maps.py", exc_info=e)
    #
    # adapted from https://stackoverflow.com/users/6099211/anton-ovsyannikov
    # correct for both bing and GMaps
    #

    def get_static_map_wh(
        self, lat=None, lng=None, zoom=19, sx=640, sy=640, crop_tiles=False
    ):
        # lat, lng - center
        # sx, sy - map size in pixels

        sy_cropped = (
            int(sy * 0.96) if crop_tiles else sy
        )  # cut off bottom 4% if cropping requested

        # common factor based on latitude
        lat_factor = math.cos(lat * math.pi / 180.0)

        # determine degree size
        globe_size = 256 * 2**zoom  # total earth map size in pixels at current zoom
        d_lng = sx * 360.0 / globe_size  # degrees/pixel
        d_lat = sy_cropped * 360.0 * lat_factor / globe_size  # degrees/pixel
        d_lat_for_url = sy * 360.0 * lat_factor / globe_size  # degrees/pixel

        # determine size in meters
        ground_resolution = 156543.04 * lat_factor / (2**zoom)  # meters/pixel
        d_x = sx * ground_resolution
        d_y = sy_cropped * ground_resolution

        # print("d_lat", d_lat, "d_lng", d_lng)
        return (d_lat, d_lat_for_url, d_lng, d_y, d_x)

    #
    # make_map_list:
    #
    # takes a center and radius, or bounds
    # returns a list of centers for zoom 19 scale 2 images
    #

    def make_tiles(self, bounds, overlap_percent=5, crop_tiles=False):
        south, west, north, east = [float(x) for x in bounds.split(",")]

        # width and height of total map
        w = abs(west - east)
        h = abs(south - north)
        lng = (east + west) / 2.0
        lat = (north + south) / 2.0

        # width and height of a tile as degrees, also get the meters
        h_tile, h_for_url, w_tile, meters, meters_x = self.get_static_map_wh(
            lng=lng, lat=lat, crop_tiles=crop_tiles
        )
        print(" tile: w:", w_tile, "h:", h_tile)

        # how many tiles horizontally and vertically?
        nx = math.ceil(w / w_tile / (1 - overlap_percent / 100.0))
        ny = math.ceil(h / h_tile / (1 - overlap_percent / 100.0))

        # now make a list of centerpoints of the tiles for the map
        tiles = []
        for row in range(ny):
            for col in range(nx):
                tiles.append(
                    {
                        "lat": north
                        - (0.5 + row) * h_tile * (1 - overlap_percent / 100.0),
                        "lat_for_url": north
                        - (0.5 * h_for_url + row * h_tile)
                        * (1 - overlap_percent / 100.0),
                        "lng": west
                        + (col + 0.5) * w_tile * (1 - overlap_percent / 100.0),
                        "h": h_tile,
                        "w": w_tile,
                        "id": len(tiles),
                        "col": col,
                        "row": row,
                    }
                )

        return tiles, nx, ny, meters, h_tile, w_tile


#
#  async file download helpers
#


def vector_tile_to_geojson(vector_tile_data):
    # Decode the vector tile
    tile = mapbox_vector_tile.decode(vector_tile_data)

    features = []
    for layer_name, layer in tile.items():
        for feature in layer["features"]:
            # Convert each feature to GeoJSON
            geojson = mapbox_vector_tile.feature_to_geojson(feature, layer_name)
            features.append(geojson)

    return {"type": "FeatureCollection", "features": features}


def convert_to_data_uri(image_content):
    image = Image.open(BytesIO(image_content))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


async def gather_urls(urls, fname, metadata, mapType, tilesMetaData, unique_directory, self):
    try:
        # execute
         # Start performance logging
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await fetch_all(
                session,
                urls,
                fname,
                metadata,
                mapType,
                unique_directory,
                tilesMetaData
            )
            # End performance logging
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Gathering URLs took {elapsed_time:.2f} seconds.")
        
    except Exception as e:
        logging.error("Error at %s", "gather_urls ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "gather_urls ts_maps.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "gather_urls ts_maps.py", exc_info=e) 

async def rate_limited_fetch(
    session, url, fname, i, mapType, unique_directory, tile, blob_service_client
):
    # if ( mapType == 'azure'):
        #Azure has more payload. Instead of increasing the sleep time exponentially, by changing it to 1 second per tile 
        # has avoided the error aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed. This change will not have
        # much impact on the time factor, especially with large number of tiles, when compared with exponentially increasing the time based on the
        # number of tiles
    try:
        # await asyncio.sleep(index * (1/50))
        # else:
        await asyncio.sleep(i * (1 / 10))
        await fetch(session, url, fname, i, mapType, unique_directory, tile, blob_service_client)
    except Exception as e:
        logging.error("Error at %s", "rate_limited_fetch ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "rate_limited_fetch ts_maps.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "rate_limited_fetch ts_maps.py", exc_info=e)  


async def fetch(session, url, fname, i, mapType, unique_directory, tile, blob_service_client):
    try:
        meta = False
        if url.endswith(" (meta)"):
            url = url[0:-7]
            meta = True

        async with session.get(url,timeout=600) as response:
            if response.status != 200:
                print(f"Error: HTTP status code {response.status}")
                error_text = await response.text()
                print(f"Error response: {error_text}")
                # response.raise_for_status()
            # #Extract metadata keys from the tile to append to the image
            tileMetadata = getTileMetaData(tile, mapType)

            # write the file
            filename = unique_directory+"/"+fname+str(i)+(".meta.txt" if meta else ".jpeg")
            blobname = fname+str(i)+(".meta.txt" if meta else ".jpeg")

            if (
                response.headers.get("Content-Type", "").lower()
                == "application/vnd.mapbox-vector-tile"
            ):
                #    async with aiofiles.open(filename, mode='rb') as f:
                #         azurevector_tile_data = await response.read()
                #         azurevectortogeojson = vector_tile_to_geojson(azurevector_tile_data)
                # print("Printing Content type " + response.headers.get('Content-Type', '').lower())
                tile_data = await response.read()
                tile = mapbox_vector_tile.decode(tile_data)
                # print(json.dumps(tile, indent=2))
                json_data = {}
                for layer_name, layer in tile.items():
                    json_data[layer_name] = {"features": []}
                    for feature in layer["features"]:
                        json_data[layer_name]["features"].append(
                            {
                                "geometry": feature["geometry"],
                                "properties": feature["properties"],
                            }
                        )
                with open(filename, "w") as f:
                    json.dump(json_data, f)
                    f.close()

            else:
                # Create a unique container
                imageConfigFile = config_dir + "/config.imagedirectory.json"
                with open(imageConfigFile, "r") as file:
                    data = json.load(file)  # Load the JSON data into a Python dictionary

                    upload_dir = data["upload_dir"]
                directoryname = upload_dir + unique_directory

                content = await response.read()

                # Code to write to Temp directory - This is required for now as there are other processes using these files
                # Need to change all the processes to read from the AIX Team's container later
                # async with aiofiles.open(filename, mode='wb') as f:
                content_type = response.headers.get("Content-Type", "")
                if (content_type.startswith("image/")) and (
                    Image.open(BytesIO(content)).mode != "RGB"
                ):
                    content = await response.read()
                    # converting to RGB
                    rgbimg = Image.open(BytesIO(content)).convert("RGB")
                    # append metadata to image
                    contentmeta = appendMetadatatoImg(rgbimg, tileMetadata, mapType)
                    blob_url = asyncio.create_task(
                        uploadImagetodirUnqFileName(
                            contentmeta, "ddphss-csels", directoryname, blobname, blob_service_client
                        )
                    )
                    # await f.write(contentmeta)
                    # await f.close()
                else:
                    if content_type.startswith("image/"):

                        content = await response.read()
                        # convert response to image
                        imgobject = Image.open(io.BytesIO(content))

                        # append metadata to image
                        contentmeta = appendMetadatatoImg(imgobject, tileMetadata, mapType)
                        blob_url = asyncio.create_task(
                            uploadImagetodirUnqFileName(
                                contentmeta, "ddphss-csels", directoryname, blobname, blob_service_client
                            )
                        )
                        # await f.write(contentmeta)
                        # await f.close()
                    else:
                        # Code to add the .txt file
                        # Adding code to write to the AIX container directory
                        blob_url = asyncio.create_task(
                            uploadImagetodirUnqFileName(
                                content, "ddphss-csels", directoryname, blobname
                            )
                        )
    except Exception as e:
        logging.error("Error at %s", "fetch ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "fetch ts_maps.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "fetch ts_maps.py", exc_info=e)            

async def fetchimagemetadata(session, fname, i, mapType, unique_directory, tile, blob_service_client):
    try:
      
        # write the file
        blobname = fname+str(i)+(".meta.txt")

        # Get the unique container
        imageConfigFile = config_dir + "/config.imagedirectory.json"
        with open(imageConfigFile, "r") as file:
            data = json.load(file)  # Load the JSON data into a Python dictionary

            upload_dir = data["upload_dir"]
            directoryname = upload_dir + unique_directory

        content = json.dumps(getTileMetaDatatoUpload(tile,mapType))

        # Code to add the .txt file
            
        blob_url = asyncio.create_task(
            uploadImagetodirUnqFileName(
                content, "ddphss-csels", directoryname, blobname, blob_service_client
            )
        )
    except Exception as e:
        logging.error("Error at %s", "fetchimagemetadata ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "fetchimagemetadata ts_maps.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "fetchimagemetadata ts_maps.py", exc_info=e)            


async def fetch_all(
    session, urls, fname, metadata, mapType, unique_directory, tilesMetaData
):
    try:
        blob_service_client = createBlobServiceClient()
        tasks = []
        for (i, tile) in enumerate(tilesMetaData):
            task = rate_limited_fetch(
                session, tile["url"], fname, i, mapType, unique_directory, tile, blob_service_client
            )
            tasks.append(task)
            # # Upload metadata .txt
            # # if metadata:
            # task = fetchimagemetadata(
            #     session,
            #     fname,
            #     i,
            #     mapType,
            #     unique_directory,
            #     tile,
            #     blob_service_client
            #     )
            # tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        logging.error("Error at %s", "fetch_all ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "fetch_all ts_maps.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "fetch_all ts_maps.py", exc_info=e)

def createBlobServiceClient():
    # Replace with your Azure Storage account details
    # Get the token using DefaultAzureCredential
        credential = DefaultAzureCredential()

        # Initialize the BlobServiceClient
        storage_account_name = "davsynapseanalyticsdev"
        
        storageconnectionstring = ts_secrets.devSecrets.getSecret(
            "TOWERSCOUTDEVSTORAGECONNSTR"
        )
        blob_service_client = BlobServiceClient.from_connection_string(
            storageconnectionstring
        )
        return blob_service_client
#
# radian conversion and Haversine distance
#


def rad(x):
    return x * math.pi / 180.0


def get_distance(x1, y1, x2, y2):
    R = 6378137.0
    # Earth’s mean radius in meters
    dLat = rad(abs(y2 - y1))
    dLong = rad(abs(x2 - x1))
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(rad(y1)) * math.cos(
        rad(y2)
    ) * math.sin(dLong / 2) * math.sin(dLong / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d
    # returns the distance in meters
def getTileMetaData(tile, mapType):
    try:
        columns_to_extract = ["id", "lat", "lng", "h", "w"]
        tileMetadata = {}

        for key in columns_to_extract:
            tileMetadata[key] = tile.get(key)
        tileMetadata["mapProvider"] = mapType
        formatted_datetime = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        tileMetadata["requestDate"] = formatted_datetime
        return tileMetadata
    except Exception as e:
        logging.error("Error at %s", "getTileMetaData ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "getTileMetaData ts_maps.py", exc_info=e)

def getTileMetaDatatoUpload(tile, mapType):
    try:
        # Specify the columns you want to extract
        columns_to_extract = ["lat", "lat_for_url", "lng", "h", "w", "id", "col", "row", "url", "filename"]
        # Extract the specified columns
        # Extracting the specified columns into a new list of dictionaries
        extracted_data = {key: tile[key] for key in columns_to_extract if key in tile}
         
        extracted_data["mapType"] = mapType
        return extracted_data
    except Exception as e:
        logging.error("Error at %s", "getTilesMetaData ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "getTilesMetaData ts_maps.py", exc_info=e)


def appendMetadatatoImg(img, tileMetaData, mapType):
    try:
        # Assign image object to a local variable
        imgImage = img

        # Get the existing EXIF data from the image
        exif_data = imgImage._getexif() if hasattr(imgImage, "_getexif") else None

        if exif_data != None:
            exif_data = imgImage._getexif()
            exif_dict = piexif.load(exif_data)
            exif_dict["Exif"] = {**exif_dict["Exif"], **tileMetaData}
            # print(f"exif_dict: {exif_dict}")
        else:
            # Create a datetime object for the current time
            now = datetime.now()
            # Convert datetime to string format recognized by EXIF
            formatted_datetime = now.strftime("%Y:%m:%d %H:%M:%S")
            # Convert the custom metadata to EXIF format
            exif_dict = {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = str(tileMetaData).encode(
                "utf-8"
            )
            
        # Encode the EXIF data as bytes
        exif_bytes = piexif.dump(exif_dict)
        #
        exif_bytes = piexif.dump(exif_dict)
        # Save updated metadata back to a byte stream using BytesIO
        # Save the modified image with updated EXIF data to a variable
        image_bytes = io.BytesIO()
        imgImage.save(image_bytes, format="JPEG", exif=exif_bytes)
        # Get the byte data from BytesIO object
        return image_bytes.getvalue()
    except Exception as e:
        logging.error("Error at %s", "appendMetaDatatoImg ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "appendMetaDatatoImg ts_maps.py", exc_info=e)

def check_bounds(x1, y1, x2, y2, bounds):
    south, west, north, east = [float(x) for x in bounds.split(",")]
    return not (y1 < south or y2 > north or x2 < west or x1 > east)


def check_tile_against_bounds(t, bounds):
    south, west, north, east = [float(x) for x in bounds.split(",")]
    x1 = t["lng"] - t["w"] / 2
    x2 = t["lng"] + t["w"] / 2
    y1 = t["lat"] + t["h"] / 2
    y2 = t["lat"] - t["h"] / 2

    return not (y1 < south or y2 > north or x2 < west or x1 > east)


async def uploadImagetodirUnqFileName(
    blobcontent, containername, directoryname, filename, blob_service_client
):
    try:
        
        container_name = containername
        container_client = blob_service_client.get_container_client(container_name)

        # Define the directory where the file will be uploaded
        directory_name = directoryname

        # Generate a unique file name using UUID
        unique_file_name = f"{uuid.uuid4()}_{filename}"
        # Create a BlobClient for the unique file in the directory
        blob_name = f"{directory_name}{unique_file_name}"
        blob_client = container_client.get_blob_client(blob_name)

        blob_client.upload_blob(blobcontent, overwrite=True, timeout=600)
        return blob_client.url
    except Exception as e:
        logging.error("Error at %s", "uploadImagetodirUnqFileName ts_maps.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "uploadImagetodirUnqFileName ts_maps.py", exc_info=e)


def generate_unique_directory_name(self):
    self.request_id = str(uuid.uuid4())[:8]  # Take first 8 characters of UUID
    return f"/{self.user_id}/{self.request_id}/"

