import pandas as pd
import get_iso_local

# get csv
df = pd.read_csv('./MTA_Subway_Entrances_and_Exits__2024_20250404.csv', delimiter=',')  # 需要根据实际情况调整分隔符

print(df.columns)
df = df[['Stop Name', 'Entrance Longitude', 'Entrance Latitude']]
df.columns = ['name', 'wgs84_lng', 'wgs84_lat']
# change / to _ to avoid bug
df['name'] = df['name'].str.replace('/', '_')
df['name'] = df['name'].replace('34 St - Hudson Yards', '34 St-Hudson Yards')
# df['name'] = df['name'].str.replace('/', '_')

subway_info = {}
subway_stations = df['name'].unique()

for station in subway_stations:
    station_data = df[df['name'] == station]

    entrances = station_data[['name', 'wgs84_lng', 'wgs84_lat']].copy()
    entrances['location'] = entrances.apply(lambda row: (row['wgs84_lng'], row['wgs84_lat']), axis=1)
    entrances = entrances.to_dict(orient='records')

    subway_info[station] = {'entrances': entrances}

travel_times = [5,10]
network_types = ['walk']
city_name = "Newyork"

false_station = []
for station in subway_info.keys():
    try:
        get_iso_local.get_iso(city_name, station, network_types, travel_times, subway_info[station])
    except Exception as e:
        print(f"Error processing {station}: {e}")
        false_station.append(station)
