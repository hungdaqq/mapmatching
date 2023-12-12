from mappymatch.constructs.trace import Trace
from mappymatch.utils.plot import plot_trace, plot_path
from mappymatch.utils.crs import LATLON_CRS, XY_CRS
from mappymatch.constructs.geofence import Geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from itertools import chain

import json
import pandas as pd
import time

from pyproj import Transformer
from dotenv import load_dotenv, find_dotenv

def xy_to_latlon(coord):

    x,y = coord
    transformer = Transformer.from_crs(XY_CRS, LATLON_CRS)
    lat, lon = transformer.transform(x, y)

    return lat, lon

import os
load_dotenv(find_dotenv())
GEOFENCE = os.environ.get("GEOFENCE")
NETWORK_TYPE = os.environ.get("NETWORK_TYPE")

# match_result.matches
class Mapmatching:
    def __init__(self, trace_path=None, geofence=1e3, network_type=NetworkType.ALL):
        self.trace = Trace.from_geojson(trace_path, index_property="coordinates", xy=True)       
        self.geofence = Geofence.from_trace(self.trace, padding=geofence) 
        self.nx_map = NxMap.from_geofence(self.geofence, network_type)

    # def plot_original_trace(self):
    #     m = plot_trace(self.trace, point_color="black", line_color="yellow")
    #     m.save("original_trace.html")

    def map_matcher(self):
        matcher = LCSSMatcher(self.nx_map)
        match_result = matcher.match_trace(self.trace)

        df = pd.DataFrame(match_result.path)

        coordinates_list = df['geom'].apply(lambda x: list(x.coords)).to_list()
        coordinates_list = list(chain(*coordinates_list))
        coordinates_list = list(map(xy_to_latlon, coordinates_list))

        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]  # GeoJSON uses [longitude, latitude] order
                    },
                    "properties": {
                        "prop0": str(i)
                    }  # You can add properties if needed
                }
                for i, (lat, lon) in enumerate(coordinates_list)
            ]
        }
        return geojson_data

    # def plot_final_path(self):
    #     match_result = self.map_matcher()
    #     m = plot_path(match_result.path, crs=self.trace.crs)
    #     m.save("final_path.html")

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/map")
async def webhook(data: dict):
    try:
        start_time = time.time()
        # Do something with the data
        with open("data.json", "w") as json_file:
            json.dump(data, json_file)

        mp = Mapmatching("data.json")
        end_time = time.time()
        # Calculate and log the computation time
        computation_time = end_time - start_time
        print(f"Computation time: {computation_time:.2f} seconds")
        return mp.map_matcher()

    except Exception as e:
        # Handle any exceptions
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)