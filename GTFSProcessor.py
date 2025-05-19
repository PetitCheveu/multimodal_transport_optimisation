import heapq
import os
import json
import pandas as pd
import numpy as np
import psycopg2
import folium
import requests
from geopy.distance import geodesic
from sqlalchemy import create_engine
from sklearn.neighbors import BallTree
from collections import defaultdict

class GTFSProcessor:
    """
    A class to process GTFS (General Transit Feed Specification) data, build multimodal transport graphs,
    enrich them with pedestrian and bicycle connections, and visualize optimal paths.

    Attributes:
        gtfs_dir (str): Directory where GTFS files are stored.
        db_params (dict): Database connection parameters.
        engine (sqlalchemy.Engine): SQLAlchemy engine for PostgreSQL interaction.
        graph (defaultdict): Adjacency list representing the multimodal graph.
        node_coords (dict): Dictionary of node coordinates by stop ID.
        bike_coords (dict): Coordinates of bike stations.
    """

    def __init__(self, gtfs_dir="data", db_params=None):
        """
        Initializes the GTFSProcessor by loading GTFS data and setting up the database engine.

        Args:
            gtfs_dir (str): Directory containing GTFS data files.
            db_params (dict): Parameters for database connection (optional).
        """
        self.stops_df = None
        self.stop_times_df = None
        self. trips_df = None
        self.routes_df = None
        self.bike_coords = None
        self.gtfs_dir = gtfs_dir
        self.db_params = db_params or {
            "dbname": os.getenv("DATABASE_NAME"),
            "user": os.getenv("DATABASE_USER"),
            "password": os.getenv("DATABASE_PASSWORD"),
            "host": os.getenv("DATABASE_HOST"),
            "port": int(os.getenv("DATABASE_PORT", 5432))
        }
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
        )
        self.graph = defaultdict(list)
        self.node_coords = {}
        self.load_dataframes()

    def load_dataframes(self):
        """
        Loads GTFS text files into pandas DataFrames.
        Required files: stops.txt, stop_times.txt, trips.txt, routes.txt.
        """
        self.stops_df = pd.read_csv(os.path.join(self.gtfs_dir, "stops.txt"), dtype={"stop_id": str})
        self.stop_times_df = pd.read_csv(os.path.join(self.gtfs_dir, "stop_times.txt"),
                                         dtype={"stop_id": str, "trip_id": str})

        self.trips_df = pd.read_csv(os.path.join(self.gtfs_dir, "trips.txt"))
        self.routes_df = pd.read_csv(os.path.join(self.gtfs_dir, "routes.txt"))

    def insert_into_db(self):
        """
        Inserts GTFS data into a PostgreSQL database using efficient bulk operations.
        """
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        from psycopg2.extras import execute_values

        stops_data = [
            (row.stop_id, row.stop_name, row.stop_lat, row.stop_lon, f"POINT({row.stop_lon} {row.stop_lat})")
            for _, row in self.stops_df.iterrows()
        ]
        execute_values(cursor, "INSERT INTO stops (stop_id, stop_name, stop_lat, stop_lon, location) VALUES %s ON CONFLICT DO NOTHING;", stops_data)

        trips_data = [
            (row.trip_id, row.route_id, row.service_id, row.trip_headsign, row.direction_id)
            for _, row in self.trips_df.iterrows()
        ]
        execute_values(cursor, "INSERT INTO trips (trip_id, route_id, service_id, trip_headsign, direction_id) VALUES %s ON CONFLICT DO NOTHING;", trips_data)

        stop_times_data = [
            (row.trip_id, row.stop_id, row.arrival_time, row.departure_time, row.stop_sequence, row.pickup_type, row.drop_off_type, row.get("shape_dist_traveled", None))
            for _, row in self.stop_times_df.iterrows()
        ]
        execute_values(cursor, "INSERT INTO stop_times (trip_id, stop_id, arrival_time, departure_time, stop_sequence, pickup_type, drop_off_type, shape_dist_traveled) VALUES %s ON CONFLICT DO NOTHING;", stop_times_data)

        routes_data = [
            (row.route_id, row.agency_id, row.route_short_name, row.route_long_name, row.route_type)
            for _, row in self.routes_df.iterrows()
        ]
        execute_values(cursor, "INSERT INTO routes (route_id, agency_id, route_short_name, route_long_name, route_type) VALUES %s ON CONFLICT DO NOTHING;", routes_data)

        conn.commit()
        cursor.close()
        conn.close()

    def fetch_shared_vehicle_stations(self):
        """
        Fetches bike station data from an external GBFS API and stores their coordinates.
        """
        url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_information.json"
        stations = requests.get(url).json()["data"]["stations"]
        self.bike_coords = {s["name"]: (s["lat"], s["lon"]) for s in stations}

    def build_graph_from_trips(self):
        """
        Constructs a graph from GTFS trip stop sequences, calculating distances and durations.
        Adds edges between consecutive stops within each trip.
        """
        self.node_coords = {
            row.stop_id: (row.stop_lat, row.stop_lon)
            for _, row in self.stops_df.iterrows()
        }

        if self.node_coords:
            print(f"🧭 {len(self.node_coords)} stop coordinates loaded")

        grouped = self.stop_times_df.groupby("trip_id")
        added = 0
        for trip_id, group in grouped:
            sorted_group = group.sort_values("stop_sequence")
            stops = list(sorted_group["stop_id"])
            for a, b in zip(stops, stops[1:]):
                if a == b:
                    continue
                coord_a = self.node_coords.get(a)
                coord_b = self.node_coords.get(b)
                if not coord_a or not coord_b:
                    continue
                distance_km = geodesic(coord_a, coord_b).km
                duration_min = (distance_km / 20) * 60
                already_linked = any(e["to"] == b and e["mode"] == "transport" for e in self.graph[a])
                if not already_linked:
                    self.graph[a].append({
                        "to": b,
                        "mode": "transport",
                        "distance_km": round(distance_km, 3),
                        "duration_min": round(duration_min, 2)
                    })
                    added += 1

        print(f"✅ Graph built with {len(self.graph)} nodes and {added} transport edges")

    def simplify_graph(self, distance_threshold=0.02):
        """
        Merges stops with identical names and close proximity to simplify the graph.

        Args:
            distance_threshold (float): Radius in kilometers to consider for merging.
        """
        ids = list(self.node_coords.keys())
        coords = np.radians([self.node_coords[i] for i in ids])
        tree = BallTree(coords)
        to_merge = {}

        for i, id1 in enumerate(ids):
            for j in tree.query_radius([coords[i]], r=distance_threshold / 6371)[0]:
                id2 = ids[j]
                if id1 != id2:
                    name1 = self.stops_df[self.stops_df.stop_id == id1].stop_name.values[0]
                    name2 = self.stops_df[self.stops_df.stop_id == id2].stop_name.values[0]
                    if name1 == name2:
                        to_merge[id2] = id1

        new_graph = defaultdict(list)
        for from_node, edges in self.graph.items():
            new_from = to_merge.get(from_node, from_node)
            for edge in edges:
                new_to = to_merge.get(edge["to"], edge["to"])
                if new_to != new_from:
                    new_graph[new_from].append({"to": new_to, "mode": edge["mode"], "distance_km": edge["distance_km"], "duration_min": edge["duration_min"]})

        self.graph = new_graph
        self.node_coords = {
            to_merge.get(n, n): c
            for n, c in self.node_coords.items()
            if to_merge.get(n, n) == n or n not in to_merge
        }

    def save_graph_to_json(self, path):
        """
        Saves the current graph and node coordinates to a JSON file.

        Args:
            path (str): Path to save the JSON output.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"graph": self.graph, "coords": self.node_coords}, f, indent=2)

    def load_graph_from_json(self, path):
        """
        Loads a graph and node coordinates from a JSON file.

        Args:
            path (str): Path to the JSON input file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = defaultdict(list, {k: v for k, v in data["graph"].items()})
        self.node_coords = {k: tuple(v) for k, v in data["coords"].items()}

    def enrich_with_bike_stations(self, gmaps):
        """
        Adds bike connections between stations using travel data from the Google Maps client.

        Args:
            gmaps (GoogleMapsClient): Client to fetch travel time and distance between points.
        """
        for i, (a_id, a_coord) in enumerate(self.bike_coords.items()):
            for j, (b_id, b_coord) in enumerate(self.bike_coords.items()):
                if a_id != b_id:
                    result = gmaps.get_travel_info(a_coord, b_coord, mode="bicycling")
                    if result:
                        self.graph[a_id].append({
                            "to": b_id,
                            "mode": "bike",
                            "distance_km": result["distance_km"],
                            "duration_min": result["duration_min"]
                        })
        self.node_coords.update(self.bike_coords)

    def enrich_with_pedestrian_links(self, max_dist_km=0.2, speed_kmh=4.5):
        """
        Adds walking links between nodes within a given distance threshold.

        Args:
            max_dist_km (float): Maximum distance in kilometers to consider for walking.
            speed_kmh (float): Assumed average walking speed in km/h.
        """
        ids = list(self.node_coords.keys())
        coords = np.radians([self.node_coords[i] for i in ids])
        tree = BallTree(coords)
        for i, id1 in enumerate(ids):
            for j in tree.query_radius([coords[i]], r=max_dist_km / 6371)[0]:
                id2 = ids[j]
                if id1 != id2:
                    dist = geodesic(self.node_coords[id1], self.node_coords[id2]).km
                    duration = (dist / speed_kmh) * 60
                    self.graph[id1].append({
                        "to": id2,
                        "mode": "walk",
                        "distance_km": round(dist, 3),
                        "duration_min": round(duration, 2)
                    })

    def enrich_transport_emissions_from_routes(self):
        """
        Enrich transport-mode edges in the graph with co2_kg estimates based on GTFS route_type:
        - Tramway (route_type == 0): 0.004 kg/km
        - Bus (route_type == 3): 0.113 kg/km
        Other modes remain unchanged or set to None.
        """
        route_type_map = {
            0: 0.004,  # Tramway
            3: 0.113  # Bus
        }

        # Map each trip_id to route_type
        trip_route_map = pd.merge(self.trips_df, self.routes_df, on="route_id", how="left")
        trip_to_type = dict(zip(trip_route_map.trip_id, trip_route_map.route_type))

        # Map each edge from GTFS stop_times to its trip_id
        stop_times_sorted = self.stop_times_df.sort_values(by=["trip_id", "stop_sequence"])
        trip_stop_pairs = []
        for trip_id, group in stop_times_sorted.groupby("trip_id"):
            stops = list(group["stop_id"])
            trip_stop_pairs += [(trip_id, a, b) for a, b in zip(stops, stops[1:])]

        trip_edge_map = defaultdict(list)
        for trip_id, a, b in trip_stop_pairs:
            trip_edge_map[(a, b)].append(trip_id)

        for from_node, edges in self.graph.items():
            for edge in edges:
                if edge["mode"] == "transport":
                    to_node = edge["to"]
                    trip_ids = trip_edge_map.get((from_node, to_node), [])
                    for trip_id in trip_ids:
                        route_type = trip_to_type.get(trip_id)
                        if route_type in route_type_map:
                            edge["co2_kg"] = round(
                                edge.get("distance_km", 0) * route_type_map[route_type], 5
                            )
                            break

    def visualize_shortest_path(self, start, end, allow_bike=True, allow_transport=True, visual_filename="maps_results/chemin_folium.html"):
        """
        Visualizes the shortest path between two nodes on a map using Folium.

        Args:
            start (str): Start node ID.
            end (str): End node ID.
            allow_bike (bool): Whether to include bike edges in the path.
            allow_transport (bool): Whether to include public transport edges.
            visual_filename (str): Path to save the resulting HTML map.

        Returns:
            tuple: (Total duration in minutes, list of node IDs representing the path)
        """
        queue = [(0, start, [])]
        visited = set()
        predecessors = {}
        costs = {start: 0}

        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]
            if node == end:
                break
            for edge in self.graph.get(node, []):
                if not allow_bike and edge.get("mode") == "bike":
                    continue
                if not allow_transport and edge.get("mode") == "transport":
                    continue
                duration = edge.get("duration_min")
                if duration is None:
                    duration = 0
                next_node = edge["to"]
                new_cost = cost + duration
                if next_node not in costs or new_cost < costs[next_node]:
                    costs[next_node] = new_cost
                    heapq.heappush(queue, (new_cost, next_node, path))
                    predecessors[next_node] = (node, edge)

        if end not in predecessors:
            print("❌ No path found.")
            return float("inf"), []

        current = end
        path = [end]
        while current != start:
            current = predecessors[current][0]
            path.insert(0, current)
        for node in self.graph:
            if node not in self.node_coords and node in self.stops_df["stop_id"].astype(str).values:
                stop_row = self.stops_df[self.stops_df["stop_id"].astype(str) == node]
                if not stop_row.empty:
                    self.node_coords[node] = (stop_row.iloc[0]["stop_lat"], stop_row.iloc[0]["stop_lon"])

        m = folium.Map(location=self.node_coords.get(start, [50.35, 3.52]), zoom_start=13)
        total_distance = 0
        folium.Marker(self.node_coords[start], tooltip=start, icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(self.node_coords[end], tooltip=end, icon=folium.Icon(color="red")).add_to(m)

        for i in range(len(path) - 1):
            c1 = self.node_coords.get(path[i])
            c2 = self.node_coords.get(path[i + 1])
            if c1 and c2:
                edge_data = next((e for e in self.graph.get(path[i], []) if e['to'] == path[i + 1]), {})
                mode = edge_data.get('mode', '?')
                dist = edge_data.get('distance_km', '?')
                duration = edge_data.get('duration_min', '?')
                co2 = edge_data.get('co2_kg', '?')
                total_distance += edge_data.get('distance_km', 0) or 0

                color = "gray"
                if mode == "transport":
                    color = "red"
                elif mode == "bike":
                    color = "blue"

                popup_text = f"{path[i]} → {path[i + 1]} | Mode: {mode} | Dist: {dist} km | Durée: {duration} min | CO2: {co2} kg"
                if c1 != self.node_coords.get(path[0]):
                    folium.Marker(c1, tooltip=path[i]).add_to(m)
                folium.PolyLine([c1, c2], color=color, tooltip=popup_text).add_to(m)
                print(            f"• {path[i]} → {path[i + 1]} | Mode : {mode} | Distance : {dist} km | Durée : {duration} min | CO2 : {co2} kg")
        m.save(f"{visual_filename}")
        print(f"📍 Map saved to: {visual_filename}")
        print("📏 Total estimated distance:", round(total_distance, 2), "km")
        print("⏱️ Total estimated duration:", costs[end], "min")
        print("Path:", " → ".join(path))
        return costs[end], path

    def find_nearest_accessible_node(self, coord, allow_bike=True):
        """
        Finds the nearest graph node to the given coordinates, optionally excluding bike-only nodes.

        Args:
            coord (tuple): Latitude and longitude coordinates.
            allow_bike (bool): Whether to include bike-only nodes.

        Returns:
            dict: Closest node info with keys: 'node', 'distance_km', 'duration_min'
        """
        from geopy.distance import geodesic

        if allow_bike:
            filtered_nodes = self.node_coords
        else:
            filtered_nodes = {
                node: pos for node, pos in self.node_coords.items()
                if any(edge["mode"] == "transport" for edge in self.graph.get(node, []))
            }

        if not filtered_nodes:
            print("❌ No accessible node found.")
            return None

        nearest = min(filtered_nodes.items(), key=lambda item: geodesic(coord, item[1]).km)
        dist_km = geodesic(coord, nearest[1]).km
        return {
            "node": nearest[0],
            "distance_km": round(dist_km, 3),
            "duration_min": round((dist_km / 4.5) * 60, 2)
        }

    def find_and_visualize_trip_from_coordinates(self, coord_start, coord_end, allow_bike=True):
        """
        Finds and visualizes the shortest multimodal path between two coordinate points.

        Args:
            coord_start (tuple): Starting coordinates.
            coord_end (tuple): Ending coordinates.
            allow_bike (bool): Whether to allow bicycle edges in routing.
        """
        nearest_start = self.find_nearest_accessible_node(coord_start, allow_bike)
        nearest_end = self.find_nearest_accessible_node(coord_end, allow_bike)
        if not nearest_start or not nearest_end:
            print("❌ No accessible node found for start or end.")
            return

        walk_to_start = nearest_start
        walk_from_end = nearest_end
        start_node = walk_to_start['node']
        end_node = walk_from_end['node']

        cost, path = self.visualize_shortest_path(start_node, end_node, allow_bike=allow_bike)

        if cost == float('inf'):
            print("❌ No path found between the selected graph nodes.")
            return

        print(f"🚶 Walking distance to start: {walk_to_start['distance_km']} km ({walk_to_start['duration_min']} min)")
        print(f"🚶 Walking distance from end: {walk_from_end['distance_km']} km ({walk_from_end['duration_min']} min)")
        total_time = cost + walk_to_start['duration_min'] + walk_from_end['duration_min']
        print(f"⏱️ Total estimated time (including walking): {round(total_time, 2)} min")

