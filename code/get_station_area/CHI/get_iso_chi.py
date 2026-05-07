import geopandas as gpd
from shapely.geometry import Point
from difflib import SequenceMatcher
import os
from collections import defaultdict
import get_iso_local

station_gdf = gpd.read_file('./CTA_RailStations/CTA_RailStations.shp').to_crs(epsg=4326)
exit_gdf = gpd.read_file('chicago_entrance.geojson').to_crs(epsg=4326)

stations = []
for idx, row in station_gdf.iterrows():
    stations.append({
        'station_id': row['STATION_ID'],
        'station_name': row['LONGNAME'],
        'geometry': row.geometry
    })

def similar(a, b):
    a = a.lower() if isinstance(a, str) else ''
    b = b.lower() if isinstance(b, str) else ''
    return SequenceMatcher(None, a, b).ratio() * 100

def find_station(exit_row):
    exit_name = exit_row.get('name', '')
    exit_point = exit_row.geometry

    best_match = None
    best_score = 0
    for station in stations:
        score = similar(exit_name, station['station_name'])
        if score > best_score:
            best_score = score
            best_match = station

    distances = [{'station': s, 'distance': exit_point.distance(s['geometry'])} for s in stations]
    nearest_station_info = min(distances, key=lambda x: x['distance'])
    nearest_station = nearest_station_info['station']
    nearest_distance = nearest_station_info['distance']

    if exit_name and best_score >= 80:
        match_distance = exit_point.distance(best_match['geometry'])
        if match_distance <= 0.018:
            return best_match['station_name']

    return nearest_station['station_name']

exit_gdf['station_name'] = exit_gdf.apply(find_station, axis=1)

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

for station_name, group in exit_gdf.groupby('station_name'):
    safe_name = station_name.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f'{safe_name}.geojson')
    group.to_file(out_path, driver='GeoJSON', encoding='utf-8')

print("results in ./output/")

geo_dict = defaultdict(list)
for idx, row in exit_gdf.iterrows():
    lon, lat = row.geometry.x, row.geometry.y
    geo_dict[row['station_name']].append((lon, lat))

travel_times = [5, 10]
network_types = ['walk']

false_station = []
for station_name, entrances in geo_dict.items():
    try:
        get_iso_local.get_iso('chi_final', station_name, network_types, travel_times, entrances)
    except Exception as e:
        print(f"found error when get isochrones：{station_name}")
        print(e)
        false_station.append(station_name)
