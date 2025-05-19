# Multimodal Route Planner – Valenciennes
This project implements a multimodal trip planner prototype for the city of Valenciennes, France, combining public transport (GTFS), shared bikes, and pedestrian segments into a unified routing system.

# Project Overview
The goal of this project is to provide optimized itineraries across multiple modes of transport using open datasets, and to visualize the computed paths on an interactive map.

Key features include:

- Parsing GTFS transport datasets (stops, trips, stop_times, routes)
- Fetching bike station locations via a real-time public API (GBFS)
- Constructing a unified multimodal graph:
- Public transport routes 
- Pedestrian connections between nearby stops 
- Bicycle links between shared stations 
- Computing the shortest or fastest path between two locations 
- Allowing mode restrictions (e.g., avoid bike, only walking)
- Displaying the trip on an interactive map (Folium)
- Estimating CO₂ emissions for each transport segment 
- Exporting/importing the graph for persistence

# Data Sources
[GTFS data – Transvilles](https://transport.data.gouv.fr/datasets?q=transvilles)

[Bike station API (Donkey Republic, GBFS)](https://transport.data.gouv.fr/datasets/velos-en-libre-service-du-reseau-transvilles-au-format-gbfs)

[National stop locations (transport.data.gouv.fr)](https://transport.data.gouv.fr/datasets/velos-en-libre-service-du-reseau-transvilles-au-format-gbfs)

You must add a folder named /data to the root of the project and place the GTFS data files inside it. The script will automatically parse them.
The GTFS data files should include:
- stops.txt
- trips.txt
- stop_times.txt
- routes.txt

# Tech Stack
**Python**

**Pandas** – for GTFS data parsing

**Folium** – for map generation

**Geopy** – for geodesic distance calculations

**BallTree** (scikit-learn) – for fast neighbor search

**PostgreSQL** (optional) – for data storage

**Google Maps API** – for enriched travel times (bike, walk)

# Features
- Fast and accurate Dijkstra-based pathfinding 
- Fully configurable transport modes 
- CO₂ estimation based on the following coefficients based on [Impact CO₂](https://impactco2.fr/outils/transport/):
  - 0.004 kg/km for tram 
  - 0.113 kg/km for bus 
  - 0 for walking and biking 
- JSON export/import of the graph 
- Visual route output in chemin_folium.html

# How to Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Add your Google Maps API  and database info in a .env file:

```plaintext
GOOGLE_MAPS_API_KEY=your_key_here

DATABASE_NAME=database_name
DATABASE_USER=database_user
DATABASE_PASSWORD=database_password
DATABASE_HOST=database_host
DATABASE_PORT=database_port
```

Place GTFS data inside the /data folder.

Run the processing script:

```bash
python main.py
```

# Example Scenarios Tested
Direct trip between two tram stops

Fully pedestrian trip (no transport available)

Trip between two shared bike stations

Combination of modes with walking transfers

Mode-restricted trip (no bike, only transit)

# Known Limitations
Real-time availability of transport/bikes not integrated

Only a single optimal path is computed (no alternatives)

CO₂ cost is displayed but not yet used as a decision criterion

No user interface (CLI only + static HTML maps)

# Future Improvements
Integration of GTFS-realtime and live bike availability

Multi-criteria route optimization (CO₂, comfort, simplicity)

Web/mobile interface

Real-time navigation and rerouting

Scalable to other cities via modular architecture

# License
This project is for educational and prototype purposes. Data is provided under open data licenses from respective platforms.

# Author
[Elena Beylat](https://github.com/PetitCheveu)<br>