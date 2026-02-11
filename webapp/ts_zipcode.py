#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
#
# Licensed under Apache 2.0
# (see LICENSE.TXT in the root of the repository for details)
#

# zipcode outline provider

import geopandas as gpd

class Zipcode_Provider:
    def __init__(self):
        self.gdf = gpd.read_file('data/tl_2019_us_zcta510/tl_2019_us_zcta510.shp')

    def zipcode_polygon(self, zipcode):
        zp = self.gdf[self.gdf['ZCTA5CE10']==zipcode]
        return zp['geometry'].to_json()
