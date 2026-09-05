'''
get isochrone for each station in Newyork
'''
import isochrone_mapbox
import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import osmnx as ox
import pyproj
from shapely.ops import transform
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from shapely.ops import orient

def get_iso(city_name, name, network_types, travel_times, station_info):

    output_folder_path = city_name + 'station' + '/' + name + '_isochrone'+'_update'  # create data folder
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)
    entrance_num = len(station_info['entrances'])
    nodes_list = station_info['entrances']
    print(station_info)
    # exit(0)
    print(name + 'totally ' + str(len(nodes_list)) + ' entrances')

    station_value = {}
    polygon_value = {}  #

    # get isochrone
    for network_type in network_types: # default as walk
        for travel_time in travel_times:  # 5/10 min
            geojson_file_path = os.path.join(output_folder_path, f"{city_name}_{name}_{travel_time}.geojson")
            if not os.path.exists(geojson_file_path):
                # print(geojson_file_path)
                print(travel_time)
                polygon_value[str(travel_time)] = {}
                isochrone_boundaries = []

                if network_type == 'walk':
                    travel_distance = 4.5 * 1000 / 60 * travel_time
                elif network_type == 'bike':
                    travel_distance = 15 * 1000 / 60 * travel_time
                elif network_type == 'drive':
                    travel_distance = 20 * 1000 / 60 * travel_time
                tmpstr = str(travel_time) + 'min ' + network_type + ' isochrone'
                station_value[tmpstr] = {}
                isochrone_dict = station_value[tmpstr]
                # get isochrones
                '''
                for each entrance，use "get_isochrone" to get some isochrones，get union for all isochrone of different entrances
                '''
                isochrones = []
                gdf = gpd.GeoDataFrame()

                for node in station_info['entrances']:
                    # print(node['location'])
                    boundary = isochrone_mapbox.get_isochrone(name, node['location'][0], node['location'][1], 'walking',
                                                              travel_time, network_type)
                    isochrone_boundaries.append(boundary)
                    gdf = pd.concat([gdf, gpd.GeoDataFrame(geometry=[boundary])])
                # get union
                merged_boundary = unary_union(gdf['geometry'])
                # merged_boundary = unary_union(isochrone_boundaries)
                merged_boundary = cascaded_union(isochrone_boundaries)
                merged_boundary_oriented = orient(merged_boundary, sign=1.0)
                gdf = gpd.GeoDataFrame(geometry=[merged_boundary_oriented])
                gdf['station'] = name
                geojson_file_path = os.path.join(output_folder_path, f"{city_name}_{name}_{travel_time}.geojson")
                print(geojson_file_path)
                gdf.to_file(geojson_file_path, driver='GeoJSON')
                gdf_4490 = gdf.set_crs(epsg=4326)
                shp_file_path = os.path.join(output_folder_path, f"{name}_{travel_time}.shp")
                gdf_4490.to_file(shp_file_path)

                print(name + 'geojson successful')
            else:
                print('exist')

        return station_value, entrance_num, output_folder_path, network_type

