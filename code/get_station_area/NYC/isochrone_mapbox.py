# Make sure to register Mapbox token before getting isochrones.

import requests
from shapely.geometry import shape
import osmnx as ox
import geopandas as gpd

def get_isochrone(name, lon, lat, transit, time, network_type):
    print('get {} iso'.format(name))
    url = 'https://api.mapbox.com/isochrone/v1/mapbox/{}/{},{}?contours_minutes=5,10,15&polygons=true&denoise=1&generalize=0&access_token=Your_token'.format(
        transit, lon, lat)
    # try:
    r = requests.get(url, timeout=(3, 7)).json()
    isochrone_5min = shape(r['features'][2]['geometry'])
    isochrone_10min = shape(r['features'][1]['geometry'])
    isochrone_15min = shape(r['features'][0]['geometry'])
    gdf = gpd.GeoDataFrame(geometry=[isochrone_5min, isochrone_10min, isochrone_15min])
    convex_hull = gdf.unary_union.convex_hull
    ISOgraph = ox.graph_from_polygon(convex_hull, network_type='all')
    fig, ax = ox.plot_graph(ISOgraph, show=False, close=False)
    print(f'{name} success！')
    if time == 5:
        gdf5 = gpd.GeoDataFrame(geometry=[isochrone_5min])
        convex_hull_5 = gdf5.unary_union.convex_hull
        isoSubgraph = ox.graph_from_polygon(convex_hull_5, network_type=network_type, retain_all=True)
        isochrone = ox.project_graph(isoSubgraph, to_crs='EPSG:4326')
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(isochrone)  # isochrone2GeoDataFrame
        boundary = gdf_nodes.unary_union.convex_hull  #
        return boundary
    if time == 10:
        gdf10 = gpd.GeoDataFrame(geometry=[isochrone_10min])
        convex_hull_10 = gdf10.unary_union.convex_hull
        isoSubgraph = ox.graph_from_polygon(convex_hull_10, network_type=network_type, retain_all=True)
        isochrone = ox.project_graph(isoSubgraph, to_crs='EPSG:4326')
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(isochrone)
        boundary = gdf_nodes.unary_union.convex_hull
        return boundary
    if time == 15:
        gdf15 = gpd.GeoDataFrame(geometry=[isochrone_15min])
        convex_hull_15 = gdf15.unary_union.convex_hull
        isoSubgraph = ox.graph_from_polygon(convex_hull_15, network_type=network_type, retain_all=True)
        isochrone = ox.project_graph(isoSubgraph, to_crs='EPSG:4326')
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(isochrone)
        boundary = gdf_nodes.unary_union.convex_hull
        return boundary
