from mappymatch.constructs.trace import Trace
from mappymatch.utils.plot import plot_trace, plot_path
from mappymatch.utils.crs import LATLON_CRS, XY_CRS
from mappymatch.constructs.geofence import Geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from itertools import chain

import pandas as pd
import json
import time

from pyproj import Transformer
from dotenv import load_dotenv, find_dotenv

import logging

# Configure the logging settings
logging.basicConfig(level=logging.INFO)  

def xy_to_latlon(coord):

    x,y = coord
    transformer = Transformer.from_crs(XY_CRS, LATLON_CRS)
    lat, lon = transformer.transform(x, y)

    return lat, lon

# import os
# load_dotenv(find_dotenv())
# GEOFENCE = os.environ.get("GEOFENCE")
# NETWORK_TYPE = os.environ.get("NETWORK_TYPE")

# match_result.matches
class Mapmatching:
    def __init__(self, trace_path=None, anomaly_data=False, geofence=1e3, network_type=NetworkType.ALL):
        self.anomaly_data = anomaly_data
        self.trace = Trace.from_geojson(trace_path, index_property="coordinates", xy=True)       
        self.geofence = Geofence.from_trace(self.trace, padding=geofence) 
        self.nx_map = NxMap.from_geofence(self.geofence, network_type)

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

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/map")
async def webhook(data: dict):
    try:
        start_time = time.time()

        with open("data.json", "w") as json_file:
            json.dump(data, json_file)
        
        logging.info("Querry license plate " + str(data["license_plate"]))
        if data['license_plate'] == license_plate:
            logging.info("MATCH")
            df = pd.DataFrame.from_dict(data)
            df['ts'] = df['features'].apply(lambda x: x['properties']['ts'])
            
            df = df[df['ts'].isin(anomaly_ts)]
            df['gps'] = df['features'].apply(lambda x: x['geometry']['coordinates'])
            # if ts.values in anomaly_ts:
            #     df = df[df['ts']==ts]
            #     print(df)
        else:
            logging.info("UNMATCH")

        mp = Mapmatching(trace_path="data.json", anomaly_data=anomaly_ts)

        end_time = time.time()
        # Calculate and log the computation time
        computation_time = end_time - start_time
        logging.info(f"Computation time: {computation_time:.2f} seconds")
        return mp.map_matcher()

    except Exception as e:
        # Handle any exceptions
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":

    anomaly_df = pd.read_csv('anomaly.csv',usecols=['latitude','longitude','plate_no','ts','labels'])
    anomaly = anomaly_df[anomaly_df['labels'] == 1]
    logging.info("Anomaly data:")
    print(anomaly)
    anomaly_ts = list(anomaly['ts'])

    license_plate = anomaly_df["plate_no"].unique()
    logging.info("Hard code with License plate " + str(license_plate[0]))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)