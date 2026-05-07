import csv

input_file = './data/NYC/entity2id.txt'
output_file = './data/NYC/Area2KG_NYCfinal.csv'

output_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            print(f"format error, skip：{line}")
            continue

        entity, kg_id = parts[0], parts[1]
        if entity.startswith('Area/'):
            area_id = entity.split('/')[1]
            output_data.append([area_id, kg_id, area_id])

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Area_id', 'KG_id', 'Region_id'])
    writer.writerows(output_data)

print(f'files have saved as {output_file}')
