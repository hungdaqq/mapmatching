from mappymatch.constructs.trace import Trace
from mappymatch.utils.crs import LATLON_CRS, XY_CRS
from mappymatch.constructs.geofence import Geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from itertools import chain

import numpy as np
import pandas as pd
import json
import time

from pyproj import Transformer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def xy_to_latlon(coord):

    x,y,anomaly = coord
    transformer = Transformer.from_crs(XY_CRS, LATLON_CRS)
    lat, lon = transformer.transform(x, y)

    return round(lat, 6), round(lon, 6), anomaly

# import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())
# GEOFENCE = os.environ.get("GEOFENCE")
# NETWORK_TYPE = os.environ.get("NETWORK_TYPE")    

class Mapmatching:
    def __init__(self, trace_path=None, anomaly_data=None, geofence=1e3, network_type=NetworkType.ALL):
        self.anomaly_data = anomaly_data
        self.trace = Trace.from_geojson(trace_path, index_property="coordinates", xy=True)       
        self.geofence = Geofence.from_trace(self.trace, padding=geofence) 
        self.nx_map = NxMap.from_geofence(self.geofence, network_type)

    def map_matcher(self):
        matcher = LCSSMatcher(self.nx_map)
        match_result = matcher.match_trace(self.trace)     

        matches_df = match_result.matches_to_dataframe()
        matches_df = matches_df[matches_df['original_lat_long'].isin(self.anomaly_data)]

        path_df = match_result.path_to_dataframe()
        path_df['road_id'] = np.where(path_df['road_id'].isin(matches_df['road_id']), 1, 0)
        path_df['geom'] = path_df.apply(lambda row: [(coord[0], coord[1], row['road_id']) for coord in row['geom']], axis=1)

        list_points = path_df['geom'].to_list()
        list_points = list(chain(*list_points))
        list_points = list(map(xy_to_latlon, list_points))

        logger.info("Returning {} points".format(len(list_points)))

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
                        "anomaly": anomaly
                    }
                }
                for (lat, lon, anomaly) in list_points
            ]
        }
        return geojson_data

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/mapmatching")
async def webhook(data: dict):
    try:
        start_time = time.time()

        with open("data.json", "w") as json_file:
            json.dump(data, json_file)
        
        querry_plate = data['features'][0]['properties']['lp']
        logger.info("Querying {} points".format(len(data['features'])))
        logger.info("Querying license plate {}".format(querry_plate))

        if querry_plate == license_plate:
            logger.info("Queried license plate MATCH hard dataset")
            data_df = pd.DataFrame.from_dict(data)
            data_df['features'] = data_df['features'].apply(lambda x: x['geometry']['coordinates'])
            data_df = data_df[data_df['features'].isin(anomaly_df)]
            anomaly_data = data_df['features'].values.tolist()
            logger.info("Found {} anomalies in query dataset".format(len(anomaly_data)))
        else:
            logger.info("Queried license plate UNMATCH hard dataset")

        mp = Mapmatching(trace_path="data.json", anomaly_data=anomaly_data)

        response = mp.map_matcher()

        # Calculate and log the computation time
        end_time = time.time()
        computation_time = end_time - start_time
        logger.info(f"Computation time: {computation_time:.2f} seconds")
        
        return response

    except Exception as e:
        # Handle any exceptions
        logger.info(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":

    anomaly_df = pd.read_csv('anomaly.csv',usecols=['latitude','longitude','plate_no','labels'])
    license_plate = anomaly_df["plate_no"].unique()
    logger.info("Hard code with License plate " + str(license_plate[0]))
    anomaly_df = anomaly_df[anomaly_df['labels'] >= 1]
    anomaly_df = anomaly_df[['longitude','latitude']].values.tolist()
    logger.info("Found {} anomalies in hard dataset".format(len(anomaly_df)))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)