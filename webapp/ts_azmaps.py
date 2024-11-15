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
   # VAULT_URL = os.environ["https://towerscout-mapkeyvault.vault.azure.net/"]
   # credential = DefaultAzureCredential()
   # client = SecretClient(vault_url=VAULT_URL, credential=credential)

   def __init__(self, api_key):
      self.key = api_key
      self.has_metadata = True
      self.mapType = "Azure"

   def get_mapkey():
      credential = DefaultAzureCredential()

      # secret_client = SecretClient(vault_url="https://towerscout-mapkeyvault.vault.azure.net/", credential=credential)
      # secretazuremapkey = secret_client.get_secret("TowerScout-Azuremapkey")

      # print(secretazuremapkey.name)
      # print(secretazuremapkey)
   
   def get_url(self, tile, zoom=19, size="640,640", sc=2, fmt="jpeg", maptype="satellite"):
      # get satellite image url for static map API
#Tiles cost less - 15 tiles 1 transaction

# curl -X GET "https://atlas.microsoft.com/map/tile?api-version=2024-04-01&tileSize=256&zoom=10&x=512&y=512&format=png&subscription-key=YourSubscriptionKey"

# https://atlas.microsoft.com/map/tile/base/{tileSetId}/{zoom}/{x}/{y}?api-version=2024-04-01&subscription-key={YourSubscriptionKey}
# https://atlas.microsoft.com/map/tile/base/microsoft.base.road/10/512/512?api-version=2024-04-01&subscription-key=YourSubscriptionKey
# https://atlas.microsoft.com/map/tile/base/microsoft.base.imagery/19/512/512?api-version=2024-04-01&subscription-key=YourSubscriptionKey
# https://atlas.microsoft.com/map/tile/base/microsoft.base.road/10/512/512?api-version=2.0&subscription-key=YourSubscriptionKey
# https://atlas.microsoft.com/map/tile/base/{tileSetId}/{zoom}/{x}/{y}?api-version=2.0&subscription-key={YourSubscriptionKey}
# curl -X GET "https://atlas.microsoft.com/map/tile/base/microsoft.base.road/10/512/512?api-version=2024-04-01&subscription-key=YourSubscriptionKey"
# https://atlas.microsoft.com/map/tile/base/{tileSetId}/{zoom}/{x}/{y}?api-version=1.0&subscription-key={YourSubscriptionKey}
# curl -X GET "https://atlas.microsoft.com/map/tile?api-version=2024-04-01&tileSize=256&zoom=10&x=512&y=512&format=png&subscription-key=YourSubscriptionKey"





    # http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/
    
   #  Do not remove - getting image with this, metadata is not working
   #  url="https://atlas.microsoft.com/map/static/png?subscription-key="+self.key+"&api-version=1.0&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+= "&height=640&width=640&tilesetId=microsoft.base&zoom=" + str(zoom)

   
   #  url="https://atlas.microsoft.com/map/static/png?subscription-key=" + self.key
   #  url+="&api-version=2022-08-01&center=" + str(tile['lat_for_url']) + "," + str(tile['lng'])
   #  url+="&zoom=19&height=640&width=640&mapstyle=satellite"


   #  url="https://atlas.microsoft.com/map/static/png?subscription-key=" + self.key
   #  url+="&api-version=2022-08-01&center=" + str(tile['lat_for_url']) + "," + str(tile['lng'])
   #  url+="&api-version=2022-08-01&center=47.6062,-122.3321&zoom=12&height=500&width=500&style=satellite"
   #  url="https://atlas.microsoft.com/map/static/png?subscription-key=" + self.key
   #  url+="&api-version=1.0&layer=hybrid&mapstyle=satellite&zoom=" + str(zoom)
   #  url+="&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&height=640&width=640&language=en"
   #  x = math.floor((tile['lng'] + 180) / 360 * math.pow(2, zoom))
      
   #  y = math.floor((1 - math.log(math.tan(tile['lat_for_url'] * math.pi / 180) + 1 / math.cos(tile['lat_for_url'] * math.pi / 180)) / math.pi) / 2 * math.pow(2, zoom))
# getting image - do not remove
   #  url="https://atlas.microsoft.com/map/imagery/png?api-version=1.0&style=satellite&zoom=19&x=" + str(x) + "&y=" + str(y)
   #  url+="&subscription-key=" + self.key

   # getting image - working - do not remove
   #  url="https://atlas.microsoft.com/map/imagery/png?api-version=1.0&style=satellite&zoom=" + str(zoom) + "&x=" + str(x) + "&y=" + str(y)
   #  url+="&subscription-key=" + self.key
   #  url+="&format=jpeg&height=700&width=700&tileSize=512"

# getting image - working - do not remove - not very clear image
   #  url="https://atlas.microsoft.com/map/tile?api-version=2022-08-01&tilesetID=microsoft.imagery&zoom=" + str(zoom)
   #  url+="&x=" + str(x) + "&y=" + str(y)
   #  url+="&subscription-key=" + self.key
   #  url+="&tileSize=512&view=auto"

   #  url="https://atlas.microsoft.com/map/tile?api-version=2022-08-01&tilesetID=microsoft.imagery&zoom=" + str(zoom)
   #  url+="&center=" + str(x) + "," + str(y)
   #  url+="&height=640&width=640"
   #  url+="&subscription-key=" + self.key
   #  url+="&tileSize=256"
   # working - do not remove
   #  url="https://atlas.microsoft.com/map/static?subscription-key=" + self.key
   #  url+="&zoom=" + str(zoom) 
   #  url+="&tilesetId=microsoft.imagery&api-version=2024-04-01&language=en-us&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&height=700&Width=700"

# working with clear image and almost close to bing - zoom 18
   #  url="https://atlas.microsoft.com/map/static?subscription-key=" + self.key
   #  url+="&zoom=" + str(zoom) 
   #  url+="&tilesetId=microsoft.imagery&api-version=2024-04-01&language=en-us&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&height=700&Width=700"

#do not remove almost close to Bing, but getting less number of tiles outside the boundary - zoom 18, height and width 640
   #  url="https://atlas.microsoft.com/map/static?subscription-key=" + self.key
   #  url+="&zoom=" + str(zoom) 
   #  url+="&tilesetId=microsoft.imagery&api-version=2024-04-01&language=en-us&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&height=640&Width=640"

# Good results - getting same results as bing- zoom = 18, height = 640 width = 640 - errors with large areas
   #  url="https://atlas.microsoft.com/map/static?subscription-key=" + self.key
   #  url+="&zoom=18" #+ str(zoom) - Need to subtract zoom level by 1 to get the same scale 
   #  url+="&tilesetId=microsoft.imagery&api-version=2024-04-01&language=en-us&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&height=640&Width=640&format=jpeg"

   #  x = math.floor((tile['lng'] + 180) / 360 * math.pow(2, zoom))
      
   #  y = math.floor((1 - math.log(math.tan(tile['lat'] * math.pi / 180) + 1 / math.cos(tile['lat'] * math.pi / 180)) / math.pi) / 2 * math.pow(2, zoom))

   #  url="https://atlas.microsoft.com/map/tile?api-version=2024-04-01&tilesetID=microsoft.imagery&zoom=19"
   #  url+="&x=" + str(x) + "&y=" + str(y)
   # #  url+="&center="+ str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&height=640&width=640"
   #  url+="&subscription-key=" + self.key
   # #  url+="&tileSize=256"

   #  url="https://atlas.microsoft.com/map/tile?api-version=2024-04-01&tilesetId=microsoft.imagery&subscription-key=" + self.key
   #  url+="&zoom=18" #+ str(zoom) - Need to subtract zoom level by 1 to get the same scale 
   #  url+="&x=" + str(x) + "&y=" + str(y)
   #  url+="&format=png"
    credential = DefaultAzureCredential()

   #  secret_client = SecretClient(vault_url="https://towerscout-mapkeyvault.vault.azure.net/", credential=credential)
   #  secretazuremapkey = secret_client.get_secret("TowerScout-Azuremapkey")

   # #  print(secretazuremapkey.name)
   #  print(secretazuremapkey)
# More results working
    url="https://atlas.microsoft.com/map/static?subscription-key=" + self.key
    url+="&zoom=18&trafficLayer=microsoft.traffic.relative.main" #+ str(zoom) - Need to subtract zoom level by 1 to get the same scale 
    url+="&tilesetId=microsoft.imagery&api-version=2024-04-01&language=en-us&center=" + str(tile['lng']) + "," + str(tile['lat_for_url'])
    url+="&height=640&Width=640&view=auto&layer=hybrid&format=jpeg&maptype=satellite&scale=2"
   #  url = f"https://atlas.microsoft.com/map/static?subscription-key={self.key}"
   #  url += f"&zoom=18"  # Adjust zoom level for better detail
   #  url += "&tilesetId=microsoft.imagery"  # Use the bird's-eye tileset
   #  url += "&api-version=2024-04-01&language=en-us"
   #  url += f"&center={str(tile['lng'])},{str(tile['lat_for_url'])}"  # Center coordinates (lng, lat)
   #  url += "&height=640&width=640&format=png-RGB&view=auto&layer=hybrid"  # Image size and format
   # # Road png
   #  url="https://atlas.microsoft.com/map/static/png?subscription-key=" + self.key
   #  url+="&api-version=1.0&layer=hybrid&zoom=" + str(zoom)
   #  url+="&center="+ str(tile['lng']) + "," + str(tile['lat_for_url'])
   #  url+="&width=800&height=600"
# Only supported in S1 pricing
   #  url = "https://atlas.microsoft.com/map/tile?subscription-key=" + self.key
   #  url+= "&api-version=1.0&tilesetId=microsoft.imagery&zoom=" + str(zoom)
   #  url+= "&x=" + str(x) + "&y=" + str(y)

   #  url = "https://atlas.microsoft.com/map/static/png?api-version=1.0&style=main&layer=basic"

   #  url+= "&zoom=" + str(zoom) + "&center="  + str(tile['lng']) + "," + str(tile['lat_for_url']) + "&width=640&height=640&format=jpeg&subscription-key=" + self.key

   #  url += str(tile['lat_for_url']) + "," + str(tile['lng']) + \
   #                "/" + str(zoom) + "?"\
   #                "&mapSize=" + size + \
   #                "&format=" + fmt + \
   #                "&subscription-key=" + self.key

   #  print(url)
    return url

   def get_meta_url(self, tile, zoom=19, size="640,640", sc=2, fmt="jpeg", maptype="satellite"):
      # https://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial/40.714550167322159,-74.007124900817871?zl=15&o=xml&key=
      #"https://atlas.microsoft.com/map/tileset/metadata/aerial"
      # url = "https://atlas.microsoft.com/map/tile/metadata/aerial?api-version=1.0"
      x = math.floor((tile['lng'] + 180) / 360 * math.pow(2, zoom))
      
      y = math.floor((1 - math.log(math.tan(tile['lat'] * math.pi / 180) + 1 / math.cos(tile['lat'] * math.pi / 180)) / math.pi) / 2 * math.pow(2, zoom))

# Working - generic metadata for all images - do not remove
      # url = "https://atlas.microsoft.com/map/tileset?api-version=2024-04-01&tilesetId=microsoft.imagery"

      url = "https://atlas.microsoft.com/map/tileset?api-version=2024-04-01&tilesetId=microsoft.imagery"
      url+= "&zoom=19&x=" + str(x) + "&y=" + str(y) + "&subscription-key=" + self.key + " (meta)"
     
      # url += str(tile['lat']) + "," + str(tile['lng']) + \
      #             "/" + str(zoom) + "?"\
      #             "&subscription-key=" + self.key + \
      #             " (meta)"
      print(url)
      return url

   #
   # checkCutOffs() 
   #
   # Function to check if the object was detected in the logo or copyright notice part
   # of the image. If so, drastically reduce confidence.
   #
   def checkCutOffs(self, object):
      if object['y2'] > 0.96 and (object['x1'] < 0.09 or object['x2'] > 0.67):
         return 0.1
      return 1

   # retrieve the data from metadata
   def get_date(self, md):
      try:
         md = json.loads(md)
         date = md['resourceSets'][0]['resources'][0]['vintageStart']
      except:
         date = "" 

      return date
   
