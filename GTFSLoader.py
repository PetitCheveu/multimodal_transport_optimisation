import pandas as pd
import os
import psycopg2
import requests
import folium
from psycopg2.extras import execute_values
from collections import defaultdict
from geopy.distance import geodesic

class GTFSLoader:
    """
    A class to load GTFS data from CSV files, insert it into a PostgreSQL database,
    fetch shared vehicle station data from Donkey API, build a graph of transit connections,
    and visualize stops and shared vehicle locations on a Folium map.
    """

    def __init__(self, gtfs_dir="data", db_params=None):
        """
        Initializes the GTFSLoader with file paths and DB connection.

        Parameters:
        - gtfs_dir (str): Directory containing GTFS CSV files.
        - db_params (dict): Dictionary of PostgreSQL connection parameters.
        """
        self.gtfs_dir = gtfs_dir
        self.db_params = db_params or {
            "dbname": "transport",
            "user": "Elena",
            "password": "E!3na2002",
            "host": "localhost",
            "port": "5432"
        }
        self.conn = psycopg2.connect(**self.db_params)
        self.cursor = self.conn.cursor()

        self.stops = None
        self.stop_times = None
        self.trips = None
        self.routes = None
        self.shared_vehicle_stations = []
        self.graph = defaultdict(list)
        self.bike_graph = defaultdict(list)
        self.walk_graph = defaultdict(list)
        self.combined_graph = defaultdict(list)

        self.station_info_url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_information.json"
        self.station_status_url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_status.json"

    def load_all(self):
        """
        Loads GTFS data from the specified directory into pandas DataFrames.
        Expected files: stops.txt, stop_times.txt, trips.txt, routes.txt.
        """
        print("📥 Chargement des fichiers GTFS...")
        try:
            self.stops = pd.read_csv(os.path.join(self.gtfs_dir, "stops.txt"))
            self.stop_times = pd.read_csv(os.path.join(self.gtfs_dir, "stop_times.txt"))
            self.trips = pd.read_csv(os.path.join(self.gtfs_dir, "trips.txt"))
            self.routes = pd.read_csv(os.path.join(self.gtfs_dir, "routes.txt"))
            print("✅ Données chargées avec succès")
        except Exception as e:
            print(f"❌ Erreur de chargement : {e}")

    def build_graph_from_gtfs(self):
        """
        Constructs a directed graph where nodes are stop_ids and edges represent
        sequential stops from the same trip_id based on stop_times.txt.
        """
        print("🔧 Construction du graphe à partir des données GTFS...")
        if self.stop_times is None:
            print("❌ Les horaires ne sont pas chargés.")
            return

        grouped = self.stop_times.groupby("trip_id")
        for trip_id, group in grouped:
            sorted_stops = group.sort_values("stop_sequence")
            stop_ids = list(sorted_stops["stop_id"])
            arrival_times = list(sorted_stops["arrival_time"])
            for i in range(len(stop_ids) - 1):
                from_stop = stop_ids[i]
                to_stop = stop_ids[i + 1]
                self.graph[from_stop].append((to_stop, trip_id, arrival_times[i], arrival_times[i + 1]))
        print(f"✅ Graphe construit avec {len(self.graph)} sommets")

    def insert_data(self):
        """
        Inserts loaded GTFS data into PostgreSQL tables: stops, trips, stop_times, routes.
        Skips records that already exist (ON CONFLICT DO NOTHING).
        """
        print("🗃️ Insertion des données en base...")
        try:
            stops_data = [
                (row.get("stop_id"), row.get("stop_name"), row.get("stop_lat"), row.get("stop_lon"), f"POINT({row.get('stop_lon')} {row.get('stop_lat')})")
                for _, row in self.stops.iterrows()
            ]
            execute_values(
                self.cursor,
                "INSERT INTO stops (stop_id, stop_name, stop_lat, stop_lon, location) VALUES %s ON CONFLICT DO NOTHING;",
                stops_data
            )

            trips_data = [
                (row.get("trip_id"), row.get("route_id"), row.get("service_id"), row.get("trip_headsign"), row.get("direction_id"))
                for _, row in self.trips.iterrows()
            ]
            execute_values(
                self.cursor,
                "INSERT INTO trips (trip_id, route_id, service_id, trip_headsign, direction_id) VALUES %s ON CONFLICT DO NOTHING;",
                trips_data
            )

            stop_times_data = [
                (row.get("trip_id"), row.get("stop_id"), row.get("arrival_time"), row.get("departure_time"), row.get("stop_sequence"), row.get("pickup_type"), row.get("drop_off_type"), row.get("shape_dist_traveled"))
                for _, row in self.stop_times.iterrows()
            ]
            execute_values(
                self.cursor,
                "INSERT INTO stop_times (trip_id, stop_id, arrival_time, departure_time, stop_sequence, pickup_type, drop_off_type, shape_dist_traveled) VALUES %s ON CONFLICT DO NOTHING;",
                stop_times_data
            )

            routes_data = [
                (row.get("route_id"), row.get("agency_id"), row.get("route_short_name"), row.get("route_long_name"), row.get("route_type"))
                for _, row in self.routes.iterrows()
            ]
            execute_values(
                self.cursor,
                "INSERT INTO routes (route_id, agency_id, route_short_name, route_long_name, route_type) VALUES %s ON CONFLICT DO NOTHING;",
                routes_data
            )

            self.conn.commit()
            print("✅ Données insérées avec succès")
        except Exception as e:
            print(f"❌ Erreur d'insertion : {e}")
            self.conn.rollback()

    def fetch_and_store_shared_vehicles(self):
        """
        Fetches shared vehicle station data from the Donkey API and inserts it
        into the PostgreSQL table shared_vehicle_stations. Also stores data locally.
        """
        print("🚴 Récupération des stations de vélos/trottinettes...")
        try:
            info_resp = requests.get(self.station_info_url)
            status_resp = requests.get(self.station_status_url)
            if info_resp.status_code == 200 and status_resp.status_code == 200:
                info_data = info_resp.json()["data"]["stations"]
                status_data = {s["station_id"]: s for s in status_resp.json()["data"]["stations"]}
                records = []
                for station in info_data:
                    sid = station["station_id"]
                    name = station["name"]
                    lat = station["lat"]
                    lon = station["lon"]
                    status = status_data.get(sid, {})
                    num_bikes = 0
                    num_scooters = 0
                    for v in status.get("vehicle_types_available", []):
                        if v["vehicle_type_id"] == "bike":
                            num_bikes = v["count"]
                        elif v["vehicle_type_id"] == "scooter":
                            num_scooters = v["count"]
                    records.append((sid, name, lat, lon, num_bikes, num_scooters, f"POINT({lon} {lat})"))
                    self.shared_vehicle_stations.append({"name": name, "lat": lat, "lon": lon})
                execute_values(
                    self.cursor,
                    "INSERT INTO shared_vehicle_stations (station_id, name, lat, lon, num_bikes, num_scooters, location) VALUES %s ON CONFLICT DO NOTHING;",
                    records
                )
                self.conn.commit()
                print(f"✅ {len(records)} stations insérées en base")
            else:
                print("❌ Erreur HTTP lors de la récupération des données vélo/trottinette")
        except Exception as e:
            print(f"❌ Erreur API vélo/trottinette : {e}")

    def generate_map(self, output_file="arrets.html"):
        """
        Generates a Folium map with GTFS stops (green) and shared vehicle stations (red).

        Parameters:
        - output_file (str): Filename for the generated HTML map.
        """
        print("🗺️ Génération de la carte des arrêts et stations...")
        map_center = [50.357, 3.525]
        fmap = folium.Map(location=map_center, zoom_start=13)

        if self.stops is not None:
            for _, row in self.stops.iterrows():
                folium.CircleMarker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    radius=4,
                    color="green",
                    fill=True,
                    fill_opacity=0.8,
                    popup=row["stop_name"]
                ).add_to(fmap)

        for station in self.shared_vehicle_stations:
            folium.Marker(
                location=[station["lat"], station["lon"]],
                popup=station["name"],
                icon=folium.Icon(color="red", icon="bicycle")
            ).add_to(fmap)

        fmap.save(output_file)
        print(f"✅ Carte enregistrée sous : {output_file}")

    def summary(self):
        """
        Displays a summary of the number of records loaded into each GTFS DataFrame.
        """
        print("\n📊 Récapitulatif :")
        if self.stops is not None:
            print(f"- {len(self.stops)} arrêts trouvés")
        if self.stop_times is not None:
            print(f"- {len(self.stop_times)} horaires trouvés")
        if self.trips is not None:
            print(f"- {len(self.trips)} trajets trouvés")
        if self.routes is not None:
            print(f"- {len(self.routes)} lignes trouvées")

        if self.shared_vehicle_stations:
            print(f"- {len(self.shared_vehicle_stations)} stations de vélos/trottinettes trouvées")

    def build_bike_graph(self, max_km=3.0, speed_kmh=18):
        """
        Builds a graph of edges between bike/trottinette stations based on geographic distance.
        Calculates travel time using an average speed.

        Parameters:
        - max_km (float): Maximum distance in kilometers to connect two stations.
        - speed_kmh (float): Speed used to calculate travel time (default 18 km/h).
        """
        print("🔧 Construction du graphe vélo entre stations...")
        for i, a in enumerate(self.shared_vehicle_stations):
            for j, b in enumerate(self.shared_vehicle_stations):
                if i != j:
                    coord_a = (a["lat"], a["lon"])
                    coord_b = (b["lat"], b["lon"])
                    dist_km = geodesic(coord_a, coord_b).km
                    if dist_km <= max_km:
                        duration_min = (dist_km / speed_kmh) * 60
                        self.bike_graph[a["name"]].append({
                            "to": b["name"],
                            "distance_km": round(dist_km, 3),
                            "duration_min": round(duration_min, 2)
                        })
        print(f"✅ Graphe vélo construit avec {len(self.bike_graph)} sommets")

    # def build_walk_graph(self, max_dist_km=0.2, speed_kmh=4.5):
    #     """
    #     Builds a pedestrian graph between all GTFS stops that are within max_dist_km
    #     of each other. Adds distance and walking time as weights.
    #
    #     Parameters:
    #     - max_dist_km (float): Maximum distance in kilometers to link stops
    #     - speed_kmh (float): Walking speed in kilometers per hour
    #     """
    #     print("🚶 Création du graphe piéton entre arrêts...")
    #     self.walk_graph = defaultdict(list)
    #     stops = self.stops.copy()
    #
    #     for i, stop_a in stops.iterrows():
    #         coord_a = (stop_a["stop_lat"], stop_a["stop_lon"])
    #         for j, stop_b in stops.iterrows():
    #             if i != j:
    #                 coord_b = (stop_b["stop_lat"], stop_b["stop_lon"])
    #                 dist_km = geodesic(coord_a, coord_b).km
    #                 if dist_km <= max_dist_km:
    #                     duration_min = (dist_km / speed_kmh) * 60
    #                     self.walk_graph[stop_a["stop_id"]].append({
    #                         "to_stop_id": stop_b["stop_id"],
    #                         "distance_km": round(dist_km, 3),
    #                         "duration_min": round(duration_min, 2)
    #                     })
    #
    #     print(f"✅ Graphe piéton construit avec {len(self.walk_graph)} sommets")

    def build_walk_graph(self, max_dist_km=0.2, speed_kmh=4.5):
        """
        Builds a pedestrian graph efficiently using BallTree between GTFS stops
        that are within max_dist_km of each other. Adds distance and walking time.
        """
        from sklearn.neighbors import BallTree
        import numpy as np

        print("🚶 Création du graphe piéton optimisé avec BallTree...")
        self.walk_graph = defaultdict(list)

        coords = np.radians(self.stops[["stop_lat", "stop_lon"]].values)
        stop_ids = self.stops["stop_id"].values
        tree = BallTree(coords, metric="haversine")
        radius = max_dist_km / 6371  # Rayon terrestre en km → radians

        for idx, (stop_id, coord) in enumerate(zip(stop_ids, coords)):
            indices = tree.query_radius([coord], r=radius, return_distance=True)
            neighbors = indices[0][0]
            distances = indices[1][0] * 6371  # radians → km

            for i, neighbor_idx in enumerate(neighbors):
                if neighbor_idx == idx:
                    continue
                to_stop_id = stop_ids[neighbor_idx]
                dist_km = round(distances[i], 3)
                duration_min = round((dist_km / speed_kmh) * 60, 2)
                self.walk_graph[stop_id].append({
                    "to_stop_id": to_stop_id,
                    "distance_km": dist_km,
                    "duration_min": duration_min
                })

        print(f"✅ Graphe piéton optimisé construit avec {len(self.walk_graph)} sommets")

    # def build_combined_graph(self):
    #     """
    #     Builds a unified multimodal graph:
    #     - Nodes represent either a bike station or a transport stop.
    #     - Edges connect:
    #         * Bike stations to other stations (mode: 'bike')
    #         * Transport stops to other stops on the same trip (mode: 'transport')
    #         * Transport stops to nearby transport stops within 200m (mode: 'walk')
    #     Each edge includes mode, distance, duration.
    #     Each node is annotated with its type: 'bike' or 'transport'.
    #     """
    #     print("🔗 Création du graphe multimodal unifié...")
    #     self.combined_graph = defaultdict(list)
    #     self.node_types = {}  # node_id -> 'transport' or 'bike'
    #
    #     # Add bike stations
    #     for station in self.shared_vehicle_stations:
    #         station_id = station["name"]
    #         self.node_types[station_id] = "bike"
    #         coord_a = (station["lat"], station["lon"])
    #         for other in self.shared_vehicle_stations:
    #             other_id = other["name"]
    #             if station_id != other_id:
    #                 coord_b = (other["lat"], other["lon"])
    #                 dist_km = geodesic(coord_a, coord_b).km
    #                 duration_min = (dist_km / 18) * 60
    #                 self.combined_graph[station_id].append({
    #                     "to": other_id,
    #                     "mode": "bike",
    #                     "distance_km": round(dist_km, 3),
    #                     "duration_min": round(duration_min, 2)
    #                 })
    #
    #     # Add transport stops
    #     for _, stop in self.stops.iterrows():
    #         stop_id = stop["stop_id"]
    #         self.node_types[stop_id] = "transport"
    #
    #     # Transport trip links
    #     for trip_id, group in self.stop_times.groupby("trip_id"):
    #         ordered = group.sort_values("stop_sequence")
    #         for i in range(len(ordered) - 1):
    #             a = ordered.iloc[i]
    #             b = ordered.iloc[i + 1]
    #             from_stop = a["stop_id"]
    #             to_stop = b["stop_id"]
    #             rows_a = self.stops[self.stops["stop_id"] == from_stop]
    #             rows_b = self.stops[self.stops["stop_id"] == to_stop]
    #             if rows_a.empty or rows_b.empty:
    #                 continue
    #             coord_a = rows_a.iloc[0]
    #             coord_b = rows_b.iloc[0]
    #             distance = geodesic((coord_a["stop_lat"], coord_a["stop_lon"]),
    #                                 (coord_b["stop_lat"], coord_b["stop_lon"])).km
    #             duration = (distance / 20) * 60  # transport average speed ~20 km/h
    #             self.combined_graph[from_stop].append({
    #                 "to": to_stop,
    #                 "mode": "transport",
    #                 "distance_km": round(distance, 3),
    #                 "duration_min": round(duration, 2),
    #                 "trip_id": trip_id
    #             })
    #
    #     # Walk links between nearby transport stops
    #     for i, stop_a in self.stops.iterrows():
    #         id_a = stop_a["stop_id"]
    #         coord_a = (stop_a["stop_lat"], stop_a["stop_lon"])
    #         for j, stop_b in self.stops.iterrows():
    #             id_b = stop_b["stop_id"]
    #             if id_a != id_b:
    #                 coord_b = (stop_b["stop_lat"], stop_b["stop_lon"])
    #                 dist_km = geodesic(coord_a, coord_b).km
    #                 if dist_km <= 0.2:
    #                     duration_min = (dist_km / 4.5) * 60
    #                     self.combined_graph[id_a].append({
    #                         "to": id_b,
    #                         "mode": "walk",
    #                         "distance_km": round(dist_km, 3),
    #                         "duration_min": round(duration_min, 2)
    #                     })
    #
    #     print(f"✅ Graphe multimodal construit avec {len(self.combined_graph)} sommets")

    def build_combined_graph(self):
        from sklearn.neighbors import BallTree
        import numpy as np

        print("🔗 Création du graphe multimodal unifié...")
        self.combined_graph = defaultdict(list)
        self.node_types = {}  # node_id -> 'transport' or 'bike'

        # Add bike stations
        for station in self.shared_vehicle_stations:
            station_id = station["name"]
            self.node_types[station_id] = "bike"
            coord_a = (station["lat"], station["lon"])
            for other in self.shared_vehicle_stations:
                other_id = other["name"]
                if station_id != other_id:
                    coord_b = (other["lat"], other["lon"])
                    dist_km = geodesic(coord_a, coord_b).km
                    duration_min = (dist_km / 18) * 60
                    self.combined_graph[station_id].append({
                        "to": other_id,
                        "mode": "bike",
                        "distance_km": round(dist_km, 3),
                        "duration_min": round(duration_min, 2)
                    })

        # Add transport stops
        for _, stop in self.stops.iterrows():
            stop_id = stop["stop_id"]
            self.node_types[stop_id] = "transport"

        # Transport trip links (unique stop sequences only)
        unique_sequences = self.stop_times.drop_duplicates(subset=["trip_id", "stop_sequence"])
        for trip_id, group in unique_sequences.groupby("trip_id"):
            ordered = group.sort_values("stop_sequence")
            stop_ids = ordered["stop_id"].tolist()
            unique_pairs = list(zip(stop_ids, stop_ids[1:]))

            for from_stop, to_stop in unique_pairs:
                rows_a = self.stops[self.stops["stop_id"] == from_stop]
                rows_b = self.stops[self.stops["stop_id"] == to_stop]
                if rows_a.empty or rows_b.empty:
                    continue
                coord_a = rows_a.iloc[0]
                coord_b = rows_b.iloc[0]
                distance = geodesic((coord_a["stop_lat"], coord_a["stop_lon"]),
                                    (coord_b["stop_lat"], coord_b["stop_lon"])).km
                duration = (distance / 20) * 60
                self.combined_graph[from_stop].append({
                    "to": to_stop,
                    "mode": "transport",
                    "distance_km": round(distance, 3),
                    "duration_min": round(duration, 2),
                    "trip_id": trip_id
                })

        # Walk links between nearby transport stops using BallTree
        coords = np.radians(self.stops[["stop_lat", "stop_lon"]].values)
        stop_ids = self.stops["stop_id"].values
        tree = BallTree(coords, metric="haversine")
        radius = 0.2 / 6371  # 200 meters in radians

        for idx, (stop_id, coord) in enumerate(zip(stop_ids, coords)):
            indices = tree.query_radius([coord], r=radius, return_distance=True)
            neighbors = indices[0][0]
            distances = indices[1][0] * 6371

            for i, neighbor_idx in enumerate(neighbors):
                if neighbor_idx == idx:
                    continue
                to_stop_id = stop_ids[neighbor_idx]
                dist_km = round(distances[i], 3)
                duration_min = round((dist_km / 4.5) * 60, 2)
                self.combined_graph[stop_id].append({
                    "to": to_stop_id,
                    "mode": "walk",
                    "distance_km": dist_km,
                    "duration_min": duration_min
                })

        print(f"✅ Graphe multimodal construit avec {len(self.combined_graph)} sommets")

    def calculate_walk_time(self, coord_a, coord_b, speed_kmh=4.5):
        """
        Calculates walking distance and estimated time between two geographic points.

        Parameters:
        - coord_a (tuple): (latitude, longitude) of point A
        - coord_b (tuple): (latitude, longitude) of point B
        - speed_kmh (float): Walking speed in kilometers per hour

        Returns:
        - dict: distance in km and duration in minutes
        """
        dist_km = geodesic(coord_a, coord_b).km
        duration_min = (dist_km / speed_kmh) * 60
        return {
            "distance_km": round(dist_km, 3),
            "duration_min": round(duration_min, 2)
        }

    def find_nearest_stop(self, coord, max_dist_km=1.0):
        """
        Finds the nearest GTFS stop to the given coordinate within a specified radius.

        Parameters:
        - coord (tuple): (latitude, longitude)
        - max_dist_km (float): Maximum radius in kilometers to consider

        Returns:
        - dict or None: Stop information or None if no stop found within the radius
        """
        if self.stops is None:
            return None

        self.stops["distance"] = self.stops.apply(
            lambda row: geodesic(coord, (row["stop_lat"], row["stop_lon"])).km, axis=1
        )
        nearby_stops = self.stops[self.stops["distance"] <= max_dist_km]
        if nearby_stops.empty:
            return None
        nearest = nearby_stops.sort_values("distance").iloc[0]
        return {
            "stop_id": nearest["stop_id"],
            "stop_name": nearest["stop_name"],
            "distance_km": round(nearest["distance"], 3),
            "lat": nearest["stop_lat"],
            "lon": nearest["stop_lon"]
        }

    # def compute_trip_between_coordinates(self, coord_start, coord_end):
    #     """
    #     Computes a multimodal trip between two coordinates using walking and direct GTFS trips.
    #     """
    #     stop_a = self.find_nearest_stop(coord_start)
    #     stop_b = self.find_nearest_stop(coord_end)
    #
    #     if not stop_a or not stop_b:
    #         return {"error": "No nearby stop found for one or both coordinates."}
    #
    #     trip_found = None
    #     for trip_id, group in self.stop_times.groupby("trip_id"):
    #         stops = group.sort_values("stop_sequence")
    #         stop_ids = list(stops["stop_id"])
    #         if stop_a["stop_id"] in stop_ids and stop_b["stop_id"] in stop_ids:
    #             if stop_ids.index(stop_a["stop_id"]) < stop_ids.index(stop_b["stop_id"]):
    #                 trip_found = trip_id
    #                 break
    #
    #     walk_to = self.calculate_walk_time(coord_start, (stop_a["lat"], stop_a["lon"]))
    #     walk_from = self.calculate_walk_time((stop_b["lat"], stop_b["lon"]), coord_end)
    #
    #     if trip_found:
    #         route_row = self.trips[self.trips["trip_id"] == trip_found].merge(self.routes, on="route_id", how="left")
    #         route_name = route_row.iloc[0]["route_long_name"] if not route_row.empty else "Unknown"
    #         return {
    #             "walk_to_stop": walk_to,
    #             "from_stop": stop_a,
    #             "to_stop": stop_b,
    #             "trip_id": trip_found,
    #             "route_name": route_name,
    #             "walk_from_stop": walk_from
    #         }
    #     else:
    #         return {
    #             "walk_to_stop": walk_to,
    #             "from_stop": stop_a,
    #             "to_stop": stop_b,
    #             "trip_id": None,
    #             "walk_from_stop": walk_from,
    #             "note": "No direct GTFS trip found."
    #         }

    def compute_trip_between_coordinates(self, coord_start, coord_end, fmap=None):
        """
        Computes a multimodal trip between two coordinates by evaluating all stops
        within a 200m radius and finding a direct GTFS connection if possible.
        If no connection is found, suggests walking. Draws results on provided map or creates new one.

        Parameters:
        - coord_start: tuple (lat, lon) for start point
        - coord_end: tuple (lat, lon) for end point
        - fmap: optional Folium map object to draw on (for combined visualization)
        """
        def stops_within_radius(coord, radius_km=0.2):
            self.stops["distance"] = self.stops.apply(
                lambda row: geodesic(coord, (row["stop_lat"], row["stop_lon"])).km, axis=1
            )
            return self.stops[self.stops["distance"] <= radius_km]

        stops_a = stops_within_radius(coord_start)
        stops_b = stops_within_radius(coord_end)

        fmap = fmap or folium.Map(location=coord_start, zoom_start=13)
        folium.Marker(coord_start, icon=folium.Icon(color="blue"), popup="Départ (Rapide)").add_to(fmap)
        folium.Marker(coord_end, icon=folium.Icon(color="orange"), popup="Arrivée (Rapide)").add_to(fmap)

        if stops_a.empty or stops_b.empty:
            folium.PolyLine([coord_start, coord_end], color="gray", weight=4, popup="Rapide - À pied").add_to(fmap)
            return fmap, {"error": "No nearby stops within 200m of start or end point."}

        for _, row_a in stops_a.iterrows():
            for _, row_b in stops_b.iterrows():
                stop_a_id = row_a["stop_id"]
                stop_b_id = row_b["stop_id"]
                for trip_id, group in self.stop_times.groupby("trip_id"):
                    stops = group.sort_values("stop_sequence")
                    stop_ids = list(stops["stop_id"])
                    if stop_a_id in stop_ids and stop_b_id in stop_ids:
                        if stop_ids.index(stop_a_id) < stop_ids.index(stop_b_id):
                            walk_to = self.calculate_walk_time(coord_start, (row_a["stop_lat"], row_a["stop_lon"]))
                            walk_from = self.calculate_walk_time((row_b["stop_lat"], row_b["stop_lon"]), coord_end)
                            route_row = self.trips[self.trips["trip_id"] == trip_id].merge(self.routes, on="route_id", how="left")
                            route_name = route_row.iloc[0]["route_long_name"] if not route_row.empty else "Unknown"

                            folium.Marker((row_a["stop_lat"], row_a["stop_lon"]), popup="Départ (Rapide)", icon=folium.Icon(color="green")).add_to(fmap)
                            folium.Marker((row_b["stop_lat"], row_b["stop_lon"]), popup="Arrivée (Rapide)", icon=folium.Icon(color="green")).add_to(fmap)
                            folium.PolyLine([(row_a["stop_lat"], row_a["stop_lon"]), (row_b["stop_lat"], row_b["stop_lon"])], color="red", weight=4, popup=f"Ligne rapide: {route_name}").add_to(fmap)

                            return fmap, {
                                "walk_to_stop": walk_to,
                                "from_stop": {
                                    "stop_id": stop_a_id,
                                    "stop_name": row_a["stop_name"],
                                    "lat": row_a["stop_lat"],
                                    "lon": row_a["stop_lon"]
                                },
                                "to_stop": {
                                    "stop_id": stop_b_id,
                                    "stop_name": row_b["stop_name"],
                                    "lat": row_b["stop_lat"],
                                    "lon": row_b["stop_lon"]
                                },
                                "trip_id": trip_id,
                                "route_name": route_name,
                                "walk_from_stop": walk_from
                            }

        walk_only = self.calculate_walk_time(coord_start, coord_end)
        folium.PolyLine([coord_start, coord_end], color="gray", weight=4, popup="Rapide - À pied").add_to(fmap)
        return fmap, {
            "mode": "walk_only",
            "distance_km": walk_only["distance_km"],
            "duration_min": walk_only["duration_min"],
            "note": "No direct GTFS trip found between nearby stops."
        }


    def compute_shortest_distance_trip(self, coord_start, coord_end):
        """
        Computes the shortest GTFS trip (in terms of geodesic distance) between two
        clusters of stops around the given coordinates. Falls back to walking if no
        such trip is found. Also visualizes the result on a Folium map named 'trajet.html'.
        """
        def stops_within_radius(coord, radius_km=0.2):
            self.stops["distance"] = self.stops.apply(
                lambda row: geodesic(coord, (row["stop_lat"], row["stop_lon"])).km, axis=1
            )
            return self.stops[self.stops["distance"] <= radius_km]

        stops_a = stops_within_radius(coord_start)
        stops_b = stops_within_radius(coord_end)

        if stops_a.empty or stops_b.empty:
            return {"error": "No nearby stops within 200m of start or end point."}

        candidates = []
        for _, row_a in stops_a.iterrows():
            for _, row_b in stops_b.iterrows():
                stop_a_id = row_a["stop_id"]
                stop_b_id = row_b["stop_id"]
                for trip_id, group in self.stop_times.groupby("trip_id"):
                    stops = group.sort_values("stop_sequence")
                    stop_ids = list(stops["stop_id"])
                    if stop_a_id in stop_ids and stop_b_id in stop_ids:
                        i, j = stop_ids.index(stop_a_id), stop_ids.index(stop_b_id)
                        if i < j:
                            sub_stops = stops.iloc[i:j+1]
                            dist = 0
                            prev_coord = None
                            path_coords = []
                            for _, row in sub_stops.iterrows():
                                current_stop = self.stops[self.stops["stop_id"] == row["stop_id"]].iloc[0]
                                current_coord = (current_stop["stop_lat"], current_stop["stop_lon"])
                                path_coords.append(current_coord)
                                if prev_coord:
                                    dist += geodesic(prev_coord, current_coord).km
                                prev_coord = current_coord
                            candidates.append((trip_id, dist, row_a, row_b, path_coords))

        walk_to = self.calculate_walk_time(coord_start, (stops_a.iloc[0]["stop_lat"], stops_a.iloc[0]["stop_lon"]))
        walk_from = self.calculate_walk_time((stops_b.iloc[0]["stop_lat"], stops_b.iloc[0]["stop_lon"]), coord_end)

        fmap = folium.Map(location=coord_start, zoom_start=13)
        folium.Marker(coord_start, icon=folium.Icon(color="blue"), popup="Départ").add_to(fmap)
        folium.Marker(coord_end, icon=folium.Icon(color="orange"), popup="Arrivée").add_to(fmap)

        if candidates:
            best_trip = min(candidates, key=lambda x: x[1])
            trip_id, dist, row_a, row_b, path_coords = best_trip
            route_row = self.trips[self.trips["trip_id"] == trip_id].merge(self.routes, on="route_id", how="left")
            route_name = route_row.iloc[0]["route_long_name"] if not route_row.empty else "Unknown"

            folium.Marker((row_a["stop_lat"], row_a["stop_lon"]), popup="Arrêt Départ", icon=folium.Icon(color="green")).add_to(fmap)
            folium.Marker((row_b["stop_lat"], row_b["stop_lon"]), popup="Arrêt Arrivée", icon=folium.Icon(color="green")).add_to(fmap)
            folium.PolyLine(path_coords, color="red", weight=4, popup=f"Ligne: {route_name}").add_to(fmap)

            fmap.save("trajet.html")
            return {
                "walk_to_stop": walk_to,
                "from_stop": {
                    "stop_id": row_a["stop_id"],
                    "stop_name": row_a["stop_name"],
                    "lat": row_a["stop_lat"],
                    "lon": row_a["stop_lon"]
                },
                "to_stop": {
                    "stop_id": row_b["stop_id"],
                    "stop_name": row_b["stop_name"],
                    "lat": row_b["stop_lat"],
                    "lon": row_b["stop_lon"]
                },
                "trip_id": trip_id,
                "route_name": route_name,
                "trip_distance_km": round(dist, 3),
                "walk_from_stop": walk_from
            }

        folium.PolyLine([coord_start, coord_end], color="gray", weight=4, popup="Trajet à pied").add_to(fmap)
        fmap.save("trajet.html")
        walk_only = self.calculate_walk_time(coord_start, coord_end)
        return {
            "mode": "walk_only",
            "distance_km": walk_only["distance_km"],
            "duration_min": walk_only["duration_min"],
            "note": "No direct GTFS trip found between nearby stops."
        }

    def export_combined_graph_to_txt(self, filepath="graphe_multimodal2.txt"):
        """
        Writes the combined multimodal graph and node types to a .txt file.
        Each node is recorded with its type.
        Each edge is recorded as: from_node,to_node,mode,distance_km,duration_min[,trip_id]
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Node types\n")
            for node_id, node_type in self.node_types.items():
                f.write(f"NODE,{node_id},{node_type}\n")
            f.write("# Edges\n")
            for from_node, edges in self.combined_graph.items():
                for edge in edges:
                    line = [str(from_node), str(edge["to"]), str(edge["mode"])]
                    line += [str(edge.get("distance_km", "")), str(edge.get("duration_min", ""))]
                    if edge["mode"] == "transport":
                        line.append(str(edge.get("trip_id", "")))
                    f.write(",".join(line) + "\n")
        print(f"📝 Graphe multimodal exporté vers {filepath}")

    def import_combined_graph_from_txt(self, filepath="graphe_multimodal.txt"):
        """
        Loads a combined multimodal graph and node types from a .txt file.
        NODE lines define types. Other lines define edges.
        """
        self.combined_graph = defaultdict(list)
        self.node_types = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("NODE"):
                    _, node_id, node_type = line.split(",")
                    self.node_types[node_id] = node_type
                else:
                    parts = line.split(",")
                    if len(parts) < 5:
                        continue
                    from_node, to_node, mode = parts[0], parts[1], parts[2]
                    edge = {
                        "to": to_node,
                        "mode": mode,
                        "distance_km": float(parts[3]) if parts[3] else None,
                        "duration_min": float(parts[4]) if parts[4] else None
                    }
                    if mode == "transport" and len(parts) > 5:
                        edge["trip_id"] = parts[5]
                    self.combined_graph[from_node].append(edge)
        print(f"📥 Graphe multimodal importé depuis {filepath}")




if __name__ == "__main__":
    """
    Main block to run the GTFSLoader and test its core functionalities:
    - Load GTFS data
    - Print summary
    - Insert into DB
    - Fetch and store shared vehicle data
    - Build a transport graph
    - Generate a map of all stops and stations
    - Build a graph of bike connections
    - Test walking time between pairs of stops
    """
    loader = GTFSLoader(gtfs_dir="data")
    loader.load_all()
    # loader.summary()
    # loader.insert_data()
    loader.fetch_and_store_shared_vehicles()
    # loader.build_graph_from_gtfs()
    # loader.generate_map()
    # loader.build_bike_graph()
    # loader.build_walk_graph()
    # loader.build_combined_graph()
    # loader.export_combined_graph_to_txt("graphe_multimodal.txt")
    loader.import_combined_graph_from_txt()


    coord_start = (50.36053582035317, 3.5208171690536227)
    coord_end = (50.3246115, 3.5146953)

    print("\n🧭 Comparaison des itinéraires (rapide vs court) sur une même carte")
    # Création de la carte de base
    shared_map = folium.Map(location=coord_start, zoom_start=13)

    # Itinéraire le plus rapide
    shared_map, result_rapide = loader.compute_trip_between_coordinates(coord_start, coord_end, fmap=shared_map)

    # Itinéraire le plus court
    result_court = loader.compute_shortest_distance_trip(coord_start, coord_end)

    # Tracer le trajet court sur la même carte
    if "trip_id" in result_court:
        folium.Marker((result_court["from_stop"]["lat"], result_court["from_stop"]["lon"]), popup="Départ (Court)",
                      icon=folium.Icon(color="purple")).add_to(shared_map)
        folium.Marker((result_court["to_stop"]["lat"], result_court["to_stop"]["lon"]), popup="Arrivée (Court)",
                      icon=folium.Icon(color="purple")).add_to(shared_map)
        folium.PolyLine([
            (result_court["from_stop"]["lat"], result_court["from_stop"]["lon"]),
            (result_court["to_stop"]["lat"], result_court["to_stop"]["lon"])
        ], color="blue", weight=4, popup=f"Ligne courte: {result_court['route_name']}").add_to(shared_map)
    else:
        folium.PolyLine([coord_start, coord_end], color="black", weight=3, popup="Court - À pied").add_to(shared_map)

    print("\n📋 Détails du trajet le plus rapide :")
    if "trip_id" in result_rapide:
        print(f"• Ligne : {result_rapide['route_name']} ({result_rapide['trip_id']})")
        print(f"• Arrêts : {result_rapide['from_stop']['stop_name']} → {result_rapide['to_stop']['stop_name']}")
        print(f"• Durée marche départ : {result_rapide['walk_to_stop']['duration_min']} min")
        print(f"• Durée marche arrivée : {result_rapide['walk_from_stop']['duration_min']} min")
    else:
        print("• Aucun trajet en transport trouvé, marche uniquement.")

    print("\n📋 Détails du trajet le plus court :")
    if "trip_id" in result_court:
        print(f"• Ligne : {result_court['route_name']} ({result_court['trip_id']})")
        print(f"• Distance totale : {result_court['trip_distance_km']} km")
        print(f"• Arrêts : {result_court['from_stop']['stop_name']} → {result_court['to_stop']['stop_name']}")
        print(f"• Durée marche départ : {result_court['walk_to_stop']['duration_min']} min")
        print(f"• Durée marche arrivée : {result_court['walk_from_stop']['duration_min']} min")
    else:
        print("• Aucun trajet en transport trouvé, marche uniquement.")

    shared_map.save("trajet.html")

    print("\nFin de calcul")
