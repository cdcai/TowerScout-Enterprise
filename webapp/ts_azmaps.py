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
# azure map class
#

from ts_maps import Map
import math
import json

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


class AzureMap(Map):
    
    def __init__(self, api_key):
        self.key = api_key
        self.has_metadata = False
        self.mapType = "Azure"

    def get_mapkey():
        credential = DefaultAzureCredential()

      

    def get_url(
        self, tile, zoom=19, size="640,640", sc=2, fmt="jpeg", maptype="satellite"
    ):
        # get satellite image url for static map API
        
        credential = DefaultAzureCredential()

        #url with low payload - trying to fix error aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed
        url = "https://atlas.microsoft.com/map/static?subscription-key=" + self.key
        url += "&zoom=18"  # + str(zoom) - Need to subtract zoom level by 1 to get the same scale
        url += (
            "&tilesetId=microsoft.imagery&api-version=2024-04-01&scale=2&center="
            + str(tile["lng"])
            + ","
            + str(tile["lat_for_url"])
        )
        url += "&height=640&Width=640&format=jpeg&labels=false&showCountryBoundary=false&traffic=false&pointOfInterest=false&showRoadLabels=false"
        

        return url
       
    def get_meta_url(
        self, tile, zoom=19, size="640,640", sc=2, fmt="jpeg", maptype="satellite"
    ):
        x = math.floor((tile["lng"] + 180) / 360 * math.pow(2, zoom))

        y = math.floor(
            (
                1
                - math.log(
                    math.tan(tile["lat"] * math.pi / 180)
                    + 1 / math.cos(tile["lat"] * math.pi / 180)
                )
                / math.pi
            )
            / 2
            * math.pow(2, zoom)
        )

        # Working - generic metadata for all images - do not remove
        # url = "https://atlas.microsoft.com/map/tileset?api-version=2024-04-01&tilesetId=microsoft.imagery"

        url = "https://atlas.microsoft.com/map/tileset?api-version=2024-04-01&tilesetId=microsoft.imagery"
        url += (
            "&zoom=19&x="
            + str(x)
            + "&y="
            + str(y)
            + "&subscription-key="
            + self.key
            + " (meta)"
        )

        print(url)
        return url

    #
    # checkCutOffs()
    #
    # Function to check if the object was detected in the logo or copyright notice part
    # of the image. If so, drastically reduce confidence.
    #
    def checkCutOffs(self, object):
        if object["y2"] > 0.96 and (object["x1"] < 0.09 or object["x2"] > 0.67):
            return 0.1
        return 1

    # retrieve the data from metadata
    def get_date(self, md):
        try:
            md = json.loads(md)
            date = md["resourceSets"][0]["resources"][0]["vintageStart"]
        except:
            date = ""

        return date
