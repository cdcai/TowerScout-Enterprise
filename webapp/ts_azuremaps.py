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
import json


class AzureMap(Map):
    def __init__(self, api_key):
        self.key = api_key
        self.has_metadata = False
        self.mapType = "Azure"

    def get_url(self, tile, zoom=19, size="640,640", fmt="jpeg"):
        # get satellite image url for Azure Maps Static Image API
        url = "https://atlas.microsoft.com/map/static/png?"
        url += (
            "subscription-key="
            + self.key
            + "&api-version=1.0"
            + "&center="
            + str(tile["lng"])
            + ","
            + str(tile["lat"])
            + "&zoom="
            + str(zoom)
            + "&size="
            + size
            + "&layer=basic"
        )  # You can change this to 'satellite' if needed
        return url

    def get_meta_url(self, tile, zoom=19):
        # Azure Maps does not provide a direct metadata URL for static images
        # You can implement a different method to retrieve metadata if needed
        return None  # No metadata available for static images

    #
    # checkCutOffs()
    #
    # Function to check if the object was detected in the logo or copyright notice part
    # of the image. If so, drastically reduce confidence.
    #
    def checkCutOffs(self, object):
        # Implement logic specific to Azure Maps if needed
        return 1  # Default confidence

    # retrieve the data from metadata
    def get_date(self, md):
        # Azure Maps does not provide metadata for static images
        return ""  # No date available
