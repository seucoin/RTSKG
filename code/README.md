**RTSKG**

The source code of "RTSKG: Building a Rail Transit Station Knowledge Graph Dataset".

The `get_station_area/` folder contains the scripts for extracting isochrone-based station areas from geographic data in different cities.

The `KG_embedding&store_recommendatation` folder provides the implementation for the KGE models and station-area store recommendatation.

The `traditional_ridership_prediction/` folder contains the implementation for the traditional knowledege-enhanced ridership prediction.  

The `LLM_ridership_prediction/` folder contains the implementation for the LLM-based knowledege-enhanced ridership prediction and the prompts.  


Some code refer to the implementations of UUKG [1] and LibCity [2].


[1] Ning Y, Liu H, Wang H, et al. UUKG: unified urban knowledge graph dataset for urban spatiotemporal prediction[J]. Advances in neural information processing systems, 2023, 36: 62442-62456.

[2] Wang J, Jiang J, Jiang W, et al. Libcity: An open library for traffic prediction[C]. Proceedings of the 29th international conference on advances in geographic information systems. 2021: 145-148.
