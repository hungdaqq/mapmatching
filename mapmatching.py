from mappymatch.constructs.trace import Trace
from mappymatch.utils.crs import LATLON_CRS, XY_CRS
from mappymatch.constructs.geofence import Geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from itertools import chain

import numpy as np
import pandas as pd
import time

from pyproj import Transformer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())
# GEOFENCE = os.environ.get("GEOFENCE")
# NETWORK_TYPE = os.environ.get("NETWORK_TYPE")    

def xy_to_latlon(coord):

    x,y,anomaly = coord
    transformer = Transformer.from_crs(XY_CRS, LATLON_CRS)
    lat, lon = transformer.transform(x, y)

    return round(lat, 6), round(lon, 6), anomaly

class Mapmatching:
    def __init__(self, trace=None, anomaly_df=None, geofence=1e3, network_type=NetworkType.ALL):
        self.anomaly_df = anomaly_df
        self.trace = Trace.from_dataframe(trace, lat_column="lat", lon_column="lon", xy=True)       
        self.geofence = Geofence.from_trace(self.trace, padding=geofence) 
        self.nx_map = NxMap.from_geofence(self.geofence, network_type)

    def map_matcher(self):
        matcher = LCSSMatcher(self.nx_map)
        match_result = matcher.match_trace(self.trace)     
        matches_df = match_result.matches_to_dataframe()
        path_df = match_result.path_to_dataframe()

        if self.anomaly_df is not None:
            matches_df = pd.merge(matches_df, self.anomaly_df, on=['lat', 'lon'], how='inner') 
            path_df['road_id'] = np.where(path_df['road_id'].isin(matches_df['road_id']), 1, 0)
        else:
            path_df['road_id'] = 0

        path_df['geom'] = path_df['geom'].apply(lambda x: x.coords)
        path_df['geom'] = path_df.apply(lambda row: [(coord[0], coord[1], row['road_id']) for coord in row['geom']], axis=1)

        list_points = path_df['geom'].to_list()
        list_points = list(chain(*list_points))
        list_points = list(map(xy_to_latlon, list_points))

        logger.info("Returning {} points".format(len(list_points)))

        response = {
            "location": [
                {
                    "lat": lat,
                    "lon": lon,
                    "anomaly": anomaly
                }
                for (lat, lon, anomaly) in list_points
            ]
        }

        return response

from flask import Flask, request, jsonify

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load anomaly data
a_df = pd.read_csv('anomaly.csv', usecols=['lat', 'lon', 'plate_no', 'labels'])
license_plate = a_df["plate_no"].unique()[0]
logger.info("Hard code with License plate {}".format(license_plate))
a_df = a_df[a_df['labels'] >= 1]
a_df = a_df[['lat', 'lon']]
logger.info("Found {} anomalies in hard dataset".format(len(a_df)))

@app.route("/mapmatching", methods=["POST"])
def map_matching():
    try:
        start_time = time.time()

        data = request.get_json()
        querry_plate = data['license_plate']
        logger.info("Querying {} points".format(len(data['locations'])))
        logger.info("Querying license plate {}".format(querry_plate))

        locations = data["locations"]
        data_df = pd.DataFrame(locations)
        data_df = data_df.drop_duplicates(subset=['lat', 'lon'])
        data_df = data_df.sort_values(by='ts')

        if querry_plate == license_plate:
            logger.info("Queried license plate MATCH hard dataset")
            anomaly_df = pd.merge(data_df, a_df, on=['lat', 'lon'], how='inner')
            anomaly_df = anomaly_df.drop('ts', axis=1)
            logger.info("Found {} anomalies in query dataset".format(len(anomaly_df)))
        else:
            anomaly_df = None
            logger.info("Queried license plate UNMATCH hard dataset")

        mp = Mapmatching(trace=data_df, anomaly_df=anomaly_df)

        response = mp.map_matcher()

        # Calculate and log the computation time
        end_time = time.time()
        computation_time = end_time - start_time
        logger.info(f"Computation time: {computation_time:.2f} seconds")

        return jsonify(response)

    except Exception as e:
        # Handle any exceptions
        logger.info(f"Error: {e}")
        return jsonify({"error": "Error processing request"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899)