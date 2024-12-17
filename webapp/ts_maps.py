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

current_directory = os.getcwd()
config_dir = os.path.join(os.getcwd().replace("webapp", ""), "webapp")
timeout = aiohttp.ClientTimeout(total=60)  # Set the timeout to 60 seconds

class Map:

    def __init__(self):
        self.has_metadata = False

    def get_sat_maps(self, tiles, loop, dir, fname):
        ssl._create_default_https_context = ssl._create_unverified_context
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
        loop.run_until_complete(
            gather_urls(urls, dir, fname, self.has_metadata, self.mapType, tiles, self)
        )
        return self.has_metadata

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


async def gather_urls(urls, dir, fname, metadata, mapType, tilesMetaData, self):
    # execute
    unique_directory = generate_unique_directory_name(self)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await fetch_all(
            session,
            urls,
            dir,
            fname,
            metadata,
            mapType,
            unique_directory,
            tilesMetaData,
        )


async def rate_limited_fetch(
    session, url, dir, fname, i, index, mapType, unique_directory, tile
):
    # if ( mapType == 'azure'):
        #Azure has more payload. Instead of increasing the sleep time exponentially, by changing it to 1 second per tile 
        # has avoided the error aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed. This change will not have
        # much impact on the time factor, especially with large number of tiles, when compared with exponentially increasing the time based on the
        # number of tiles
    await asyncio.sleep(index * (1))
    # else:
    #     await asyncio.sleep(index * (1 / 3))
    await asyncio.sleep(index * (1))
    # else:
    #     await asyncio.sleep(index * (1 / 3))
    await fetch(session, url, dir, fname, i, mapType, unique_directory, tile)


async def fetch(session, url, dir, fname, i, mapType, unique_directory, tile):

    meta = False
    if url.endswith(" (meta)"):
        url = url[0:-7]
        meta = True

    async with session.get(url) as response:
        if response.status != 200:
            print(f"Error: HTTP status code {response.status}")
            error_text = await response.text()
            print(f"Error response: {error_text}")
            # response.raise_for_status()
        # #Extract metadata keys from the tile to append to the image
        tileMetadata = getTileMetaData(tile, mapType)

        # write the file
        filename = dir + "/" + fname + str(i) + (".meta.txt" if meta else ".jpeg")
        blobname = fname + str(i) + (".meta.txt" if meta else ".jpeg")

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
                        contentmeta, "ddphss-csels", directoryname, blobname
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
                            contentmeta, "ddphss-csels", directoryname, blobname
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
                    # await f.write(content)
                    # await f.close()
                # blob_url =  uploadImage(await response.read(),fname+str(i))
                # blob_url=uploadImageUnqFileName(await response.read(),fname+str(i))


async def fetch_all(
    session, urls, dir, fname, metadata, mapType, unique_directory, tilesMetaData
):
    tasks = []
    for i, tile in enumerate(tilesMetaData):
        suffix = i
        task = rate_limited_fetch(
            session, tile["url"], dir, fname, suffix, i, mapType, unique_directory, tile
        )
        tasks.append(task)
        if metadata:
            task = rate_limited_fetch(
                session,
                tile["metaurl"],
                dir,
                fname,
                suffix,
                i,
                mapType,
                unique_directory,
                tile,
            )
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


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


def appendMetadatatoPng(img, tilesMetadata):
    imgmeta = Image.open(img)
    # Create a new metadata object
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("lat", tilesMetadata["lat"])
    metadata.add_text("lat_for_url", tilesMetadata["lat_for_url"])
    metadata.add_text("lng", tilesMetadata["lng"])
    metadata.add_text("h", tilesMetadata["h"])
    metadata.add_text("w", tilesMetadata["w"])
    metadata.add_text("id", tilesMetadata["id"])
    metadata.add_text("col", tilesMetadata["col"])
    metadata.add_text("row", tilesMetadata["row"])
    metadata.add_text("url", tilesMetadata["url"])
    imgmeta.save(img, "JPEG", pnginfo=metadata)


def appendMetadatatoJpeg(img, tilesMetadata):
    # Create a BytesIO object from the image data
    # image_io = BytesIO(img)
    # Open the image
    if not img:
        print("Error: Img data is empty.")
    image = Image.open(img)
    if not image:
        print("Error: Image data is empty.")

    # Load existing EXIF data or create a new one
    exif_dict = piexif.load(image.info.get("exif", b""))

    # Prepare the new data from the dictionary
    new_data = json.dumps(tilesMetadata)

    # Check if UserComment tag exists
    user_comment_tag = piexif.ExifIFDName.UserComment
    existing_comment = exif_dict["Exif"].get(user_comment_tag)

    if existing_comment:
        # Decode the existing comment and append new data
        existing_comment = existing_comment.decode(
            "utf-8"
        )  # Decode from bytes to string
        updated_comment = existing_comment + " " + new_data
    else:
        # If the tag doesn't exist or has no value, set it to the new data
        updated_comment = new_data.strip()

    # Update the EXIF dictionary
    exif_dict["Exif"][user_comment_tag] = updated_comment.encode(
        "utf-8"
    )  # Encode back to bytes

    # Convert the updated EXIF data back to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save the updated image with new EXIF data
    image.save(img, exif=exif_bytes)
    return img


def getTileMetaData(tile, mapType):
    columns_to_extract = ["id", "lat", "lng", "h", "w"]
    tileMetadata = {}

    for key in columns_to_extract:
        tileMetadata[key] = tile.get(key)
    tileMetadata["mapProvider"] = mapType
    formatted_datetime = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    tileMetadata["requestDate"] = formatted_datetime
    return tileMetadata


def getTilesMetaData(tiles, mapType):
    # Specify the columns you want to extract
    columns_to_extract = ["id", "lat", "lng", "h", "w"]
    # Extract the specified columns
    # Extracting the specified columns into a new list of dictionaries
    extracted_data = [
        {key: item[key] for key in columns_to_extract if key in item} for item in tiles
    ]
    extracted_data["mapType"] = mapType
    return extracted_data


def appendMetadatatoImg(img, tileMetaData, mapType):
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
        # exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
        exif_dict = {"Exif": {}}
        # exif_dict["0th"][piexif.ImageIFD.DateTime] = formatted_datetime
        # exif_dict["0th"][piexif.ImageIFD.ColorMap] = mapType
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = str(tileMetaData).encode(
            "utf-8"
        )
        # print(f"exif_dict: {exif_dict}")
    # Encode the EXIF data as bytes
    exif_bytes = piexif.dump(exif_dict)
    #
    exif_bytes = piexif.dump(exif_dict)
    # print(f"exif_bytes{exif_bytes}")
    # Save updated metadata back to a byte stream using BytesIO
    # output_stream = io.BytesIO()
    # Save the modified image with updated EXIF data to a variable
    image_bytes = io.BytesIO()
    imgImage.save(image_bytes, format="JPEG", exif=exif_bytes)
    # Get the byte data from BytesIO object
    return image_bytes.getvalue()


# Reset the stream position to start
# output_stream.seek(0)

# Read the byte stream and store it in a variable
# modified_image_data = output_stream.read()

# Close the stream to free up resources
# output_stream.close()
# return modified_image_data
#
# bounds checking
#


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


def uploadImage(blbname):

    # Replace with your Azure Storage account details
    connect_str = "DefaultEndpointsProtocol=https;AccountName=tsstorageaccountvssubscr;AccountKey=l7FCoM2QC+3Xljbb9EkwdZXYxtHBkH7GQX6ta5aHZf2i6N9ZCLUSufGaVIUcD693A8cp1QIHzQDO+AStjbtSQQ==;EndpointSuffix=core.windows.net"
    container_name = "firstcontainerunderroot"
    blob_name = blbname
    file_path = "path_to_your_local_file"

    # data = b"Hello, Azure Blob Storage!"

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container_client = blob_service_client.get_container_client(container_name)

    # Create a BlobClient object for the blob
    # blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # Create a BlobClient object for the blob (file)

    # # Upload data to the blob
    # blob_client.upload_blob(data)

    local_file_path = (
        "C:/TowerScout/Testing/Bing/BingTempfolders/tmpg2scritl/tmpg2scritl1.jpg"
    )
    # blob_client = container_client.get_blob_client(blob=local_file_path.split("/")[-1])
    blob_client = container_client.get_blob_client(blob_name)
    # Upload the file
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, timeout=60)
        return blob_client.url
    # uploadimagecontent = readimagecontent(local_file_path)

    # container_client.upload_blob(name=blob_name, data=uploadimagecontent)


async def readimagecontent(local_file_path):
    async with aiofiles.open(local_file_path, "wb") as file:
        uploadimagecontent = await file.read()
        await file.close()
    return uploadimagecontent


def uploadImageUnqFileName(blobcontent, filename):

    # Replace with your Azure Storage account details
    connect_str = "DefaultEndpointsProtocol=https;AccountName=tsstorageaccountvssubscr;AccountKey=l7FCoM2QC+3Xljbb9EkwdZXYxtHBkH7GQX6ta5aHZf2i6N9ZCLUSufGaVIUcD693A8cp1QIHzQDO+AStjbtSQQ==;EndpointSuffix=core.windows.net"
    container_name = "firstcontainerunderroot"

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container_client = blob_service_client.get_container_client(container_name)

    # Define the directory where the file will be uploaded
    directory_name = "my-images-directory/"

    # Generate a unique file name using UUID
    unique_file_name = f"{uuid.uuid4()}_{filename}"
    # Create a BlobClient for the unique file in the directory
    # blob_name = f"{directory_name}{unique_file_name}"
    blob_name = f"{unique_file_name}"
    blob_client = container_client.get_blob_client(blob_name)

    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url

    # uploadimagecontent = readimagecontent(local_file_path)

    # container_client.upload_blob(name=blob_name, data=uploadimagecontent)


async def uploadImagetodirUnqFileName(
    blobcontent, containername, directoryname, filename
):

    # Replace with your Azure Storage account details
    # Get the token using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Initialize the BlobServiceClient
    storage_account_name = "davsynapseanalyticsdev"
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net/",
        credential=credential,
    )
    storageconnectionstring = ts_secrets.devSecrets.getSecret(
        "TOWERSCOUTDEVSTORAGECONNSTR"
    )
    blob_service_client = BlobServiceClient.from_connection_string(
        storageconnectionstring
    )
    container_name = containername
    container_client = blob_service_client.get_container_client(container_name)

    # Define the directory where the file will be uploaded
    directory_name = directoryname

    # Generate a unique file name using UUID
    unique_file_name = f"{uuid.uuid4()}_{filename}"
    # Create a BlobClient for the unique file in the directory
    blob_name = f"{directory_name}{unique_file_name}"
    blob_client = container_client.get_blob_client(blob_name)

    blob_client.upload_blob(blobcontent, overwrite=True, timeout=60)
    return blob_client.url

    # uploadimagecontent = readimagecontent(local_file_path)

    # container_client.upload_blob(name=blob_name, data=uploadimagecontent)


def createUnqDir(containername):

    # Replace with your Azure Storage account details
    # Get the token using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Initialize the BlobServiceClient
    storage_account_name = "davsynapseanalyticsdev"
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net/",
        credential=credential,
    )

    # Define the directory where the file will be uploaded
    base_name = "dirazuremaptiles"
    unique_directory_name = generate_unique_directory_name()

    container_name = containername
    # Create a blob client to upload a placeholder file
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=f"{unique_directory_name}placeholder.txt"
    )

    # Upload a zero-byte blob to create the directory
    blob_client.upload_blob(b"", overwrite=True)

    return unique_directory_name

    # uploadimagecontent = readimagecontent(local_file_path)

    # container_client.upload_blob(name=blob_name, data=uploadimagecontent)


def generate_unique_container_name(base_name):
    # Generate a unique name by appending the current timestamp and a UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Take first 8 characters of UUID
    return f"{base_name}-{timestamp}-{unique_id}"


def generate_unique_directory_name(self):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    self.request_id = str(uuid.uuid4())[:8]  # Take first 8 characters of UUID
    self.user_id = get_current_user()

    return f"/{self.user_id}/{self.request_id}/"


import getpass


def get_current_user():
    username = getpass.getuser()  # Get the username
    # Optionally, you can also get the domain
    domain = os.getenv("USERDOMAIN") or os.getenv("COMPUTERNAME")
    return f"{username}"


# def get_current_user():
#     username = getpass.getuser()  # Get the username
#     # Optionally, you can also get the domain
#     domain = os.getenv("USERDOMAIN") or os.getenv("COMPUTERNAME")
#     logging.info("ts_maps domain {domain} username {username}")
#     # Implementing for azure app services
#     user_id = request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
#     return f"{user_id}"


def uploadImage(blobcontent, blbname):

    # Replace with your Azure Storage account details
    connect_str = "DefaultEndpointsProtocol=https;AccountName=tsstorageaccountvssubscr;AccountKey=l7FCoM2QC+3Xljbb9EkwdZXYxtHBkH7GQX6ta5aHZf2i6N9ZCLUSufGaVIUcD693A8cp1QIHzQDO+AStjbtSQQ==;EndpointSuffix=core.windows.net"
    container_name = "firstcontainerunderroot"
    blob_name = blbname
    # file_path = "path_to_your_local_file"

    # data = b"Hello, Azure Blob Storage!"

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container_client = blob_service_client.get_container_client(container_name)

    # Create a BlobClient object for the blob
    # blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # Create a BlobClient object for the blob (file)

    # # Upload data to the blob
    # blob_client.upload_blob(data)

    # local_file_path = "C:/TowerScout/Testing/Bing/BingTempfolders/tmpg2scritl/tmpg2scritl1.jpg"
    # blob_client = container_client.get_blob_client(blob=local_file_path.split("/")[-1])
    blob_client = container_client.get_blob_client(blob_name)
    # # Upload the file
    # with open(local_file_path, "rb") as data:
    #     blob_client.upload_blob(data)

    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url

    # uploadimagecontent = readimagecontent(local_file_path)

    # container_client.upload_blob(name=blob_name, data=uploadimagecontent)


def uploadImagetoUnqDir(blobcontent, containername, directoryname, filename):

    # Replace with your Azure Storage account details
    # Get the token using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Initialize the BlobServiceClient
    storage_account_name = "davsynapseanalyticsdev"
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net/",
        credential=credential,
    )

    container_name = containername
    container_client = blob_service_client.get_container_client(container_name)

    # Define the directory where the file will be uploaded
    directory_name = directoryname

    # Create a BlobClient for the file in the directory
    blob_name = f"{directory_name}{filename}"
    blob_client = container_client.get_blob_client(blob_name)

    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url

    # uploadimagecontent = readimagecontent(local_file_path)

    # container_client.upload_blob(name=blob_name, data=uploadimagecontent)
