# RTSKG — Example SPARQL Queries for the Competency Questions

These are example SPARQL queries for the competency questions listed in the
project [README](../README.md#competency-questions).

All queries share these prefixes:

```sparql
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX cls:  <https://w3id.org/rtskg/ontology/class/>
PREFIX op:   <https://w3id.org/rtskg/ontology/objectproperty/>
PREFIX dp:   <https://w3id.org/rtskg/ontology/dataproperty/>
```

```sparql
# CQ1
SELECT ?line ?lineLabel WHERE {
  VALUES ?stationLabel { "1 Av" }
  ?station rdfs:label ?stationLabel ;
           op:station_belongs_to_line ?line .
  ?line rdfs:label ?lineLabel .
}
```

```sparql
# CQ2
SELECT ?minutes ?poi ?poiLabel WHERE {
  VALUES ?stationLabel { "1 Av" }
  ?station rdfs:label ?stationLabel .
  ?sa op:station_area_belongs_to_station ?station ;
      dp:walking_time_minutes ?minutes .
  ?poi op:poi_locates_at_station_area ?sa ;
       rdfs:label ?poiLabel .
}
ORDER BY ?minutes
```

```sparql
# CQ3
SELECT ?category (COUNT(?poi) AS ?n) WHERE {
  VALUES ?sa { <https://w3id.org/rtskg/new_york_city/instance/station_area_1> }
  ?poi op:poi_locates_at_station_area ?sa ;
       op:poi_has_poi_category ?category .
}
GROUP BY ?category ORDER BY DESC(?n) LIMIT 1
```

```sparql
# CQ4
SELECT ?category (COUNT(?poi) AS ?n) WHERE {
  VALUES ?block { <https://w3id.org/rtskg/new_york_city/instance/block_0> }
  ?poi op:poi_locates_at_block ?block ;
       op:poi_has_poi_category ?category .
}
GROUP BY ?category ORDER BY DESC(?n)
```

```sparql
# CQ5
SELECT ?road (COUNT(DISTINCT ?fa) AS ?numFunctionalAreas) WHERE {
  ?road op:road_intersects_functional_area ?fa .
}
GROUP BY ?road
HAVING (COUNT(DISTINCT ?fa) > 1)
ORDER BY DESC(?numFunctionalAreas)
```

```sparql
# CQ6
SELECT ?roadCategory (COUNT(?road) AS ?n) WHERE {
  VALUES ?faLabel { "Chinatown" }
  ?fa rdfs:label ?faLabel .
  ?road op:road_intersects_functional_area ?fa ;
        op:road_has_road_category ?roadCategory .
}
GROUP BY ?roadCategory ORDER BY DESC(?n)
```

```sparql
# CQ7
SELECT ?borough (COUNT(DISTINCT ?station) AS ?numStations) WHERE {
  VALUES ?faLabel { "Chinatown" }
  ?fa rdfs:label ?faLabel ;
      op:functional_area_locates_at_borough ?borough .
  ?sa op:station_area_locates_at_functional_area ?fa ;
      op:station_area_belongs_to_station ?station .
}
GROUP BY ?borough
```

```sparql
# CQ8
SELECT ?neighbor ?category (COUNT(?poi) AS ?n) WHERE {
  VALUES ?faLabel { "Chinatown" }
  ?fa rdfs:label ?faLabel ;
      op:functional_area_is_adjacent_to_functional_area ?neighbor .
  ?block op:block_locates_at_functional_area ?neighbor .
  ?poi op:poi_locates_at_block ?block ;
       op:poi_has_poi_category ?category .
}
GROUP BY ?neighbor ?category ORDER BY ?neighbor DESC(?n)
```

```sparql
# CQ9
SELECT DISTINCT ?block WHERE {
  VALUES ?lineLabel { "4th Av" }
  ?line rdfs:label ?lineLabel .
  ?station op:station_belongs_to_line ?line .
  ?sa op:station_area_belongs_to_station ?station ;
      op:station_area_intersects_block ?block .
}
```

```sparql
# CQ10
SELECT ?station ?category (COUNT(?poi) AS ?n) WHERE {
  VALUES ?lineLabel { "4th Av" }
  ?line rdfs:label ?lineLabel .
  ?station op:station_belongs_to_line ?line .
  ?sa op:station_area_belongs_to_station ?station ;
      dp:walking_time_minutes 5 .
  ?poi op:poi_locates_at_station_area ?sa ;
       op:poi_has_poi_category ?category .
}
GROUP BY ?station ?category ORDER BY ?station DESC(?n)
```
