import requests
import os
import psycopg2
import pandas as pd
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine
import heapq
from sklearn.neighbors import BallTree
import numpy as np
from tqdm import tqdm
import folium

class GoogleMapsClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("Google Maps API key not found in .env file under GOOGLE_MAPS_API_KEY")
        self.base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    def get_travel_info(self, origin, destination, mode="walking"):
        params = {
            "origins": f"{origin[0]},{origin[1]}",
            "destinations": f"{destination[0]},{destination[1]}",
            "mode": mode,
            "key": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            try:
                element = data["rows"][0]["elements"][0]
                if element["status"] == "OK":
                    distance_km = element["distance"]["value"] / 1000
                    duration_min = element["duration"]["value"] / 60
                    return {
                        "distance_km": round(distance_km, 3),
                        "duration_min": round(duration_min, 2)
                    }
            except (IndexError, KeyError):
                return None
        return None

    def enrich_graph_with_api(self, graph, node_coords, mode="bicycling", max_edges=100):
        print(f"📡 Enrichissement du graphe avec Google Maps API (mode: {mode})...")
        self.api_cache = {}
        count = 0
        for from_node, edges in tqdm(graph.items(), desc=f"Enrichissement {mode}"):
            coord_a = node_coords.get(from_node)
            if not coord_a:
                continue
            for edge in edges:
                to_node = edge.get("to")
                coord_b = node_coords.get(to_node)
                if not coord_b:
                    continue
                key = (from_node, to_node, mode)
                if key in self.api_cache:
                    result = self.api_cache[key]
                else:
                    result = self.get_travel_info(coord_a, coord_b, mode=mode)
                    if result:
                        self.api_cache[key] = result
                if result:
                    edge["distance_km"] = result["distance_km"]
                    edge["duration_min"] = result["duration_min"]
                    count += 1
                    if count >= max_edges:
                        print(f"⏹️ Limite atteinte ({max_edges} arêtes enrichies)")
                        return
        print("✅ Graphe enrichi avec les données API")

    def build_full_multimodal_graph(self, transport_edges, bike_stations, stop_coords, bike_coords):
        graph = {}
        node_coords = {}

        for from_stop, to_stop in transport_edges:
            graph.setdefault(from_stop, []).append({"to": to_stop, "mode": "transport"})
        node_coords.update(stop_coords)

        walk_edges = {}
        stop_ids = list(stop_coords.keys())
        stop_array = np.radians([stop_coords[sid] for sid in stop_ids])
        stop_tree = BallTree(stop_array, metric="haversine")
        stop_radius = 0.2 / 6371.0
        for idx, coord in enumerate(stop_array):
            neighbors = stop_tree.query_radius([coord], r=stop_radius)[0]
            for j in neighbors:
                if j != idx:
                    sid1, sid2 = stop_ids[idx], stop_ids[j]
                    walk_edges.setdefault(sid1, []).append({"to": sid2, "mode": "walk"})

        for sid, scoord in stop_coords.items():
            for bid, bcoord in bike_coords.items():
                graph.setdefault(sid, []).append({"to": bid, "mode": "walk"})
                graph.setdefault(bid, []).append({"to": sid, "mode": "walk"})

        bike_ids = list(bike_coords.keys())
        bike_array = np.radians([bike_coords[b] for b in bike_ids])
        bike_tree = BallTree(bike_array, metric="haversine")
        bike_radius = 10.0 / 6371.0
        for idx, coord in enumerate(bike_array):
            neighbors = bike_tree.query_radius([coord], r=bike_radius)[0]
            for j in neighbors:
                if j != idx:
                    b1, b2 = bike_ids[idx], bike_ids[j]
                    graph.setdefault(b1, []).append({"to": b2, "mode": "bike"})

        node_coords.update(bike_coords)

        self.enrich_graph_with_api(walk_edges, node_coords, mode="walking", max_edges=100)
        for from_node, edges in walk_edges.items():
            graph.setdefault(from_node, []).extend(edges)

        self.enrich_graph_with_api(graph, node_coords, mode="bicycling", max_edges=100)

        return graph, node_coords

    def save_graph_to_json(self, graph, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2)
        print(f"📁 Graphe sauvegardé dans '{filepath}'")

    def load_graph_from_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        print(f"📂 Graphe chargé depuis '{filepath}'")
        return graph

    def visualize_shortest_path(self, graph, node_coords, start, end):
        queue = [(0, start, [])]
        visited = set()
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]
            if node == end:
                m = folium.Map(location=node_coords.get(start, [50.35, 3.52]), zoom_start=13)
                for i in range(len(path)-1):
                    p1, p2 = path[i], path[i+1]
                    c1 = node_coords.get(p1)
                    c2 = node_coords.get(p2)
                    if c1 and c2:
                        folium.Marker(c1, tooltip=p1).add_to(m)
                        folium.PolyLine([c1, c2], color="blue").add_to(m)
                if node_coords.get(end):
                    folium.Marker(node_coords[end], tooltip=end, icon=folium.Icon(color="green")).add_to(m)
                m.save("chemin_folium.html")
                print("📍 Carte enregistrée : chemin_folium.html")
                return cost, path
            for edge in graph.get(node, []):
                next_node = edge["to"]
                duration = edge.get("duration_min")
                if duration is not None:
                    heapq.heappush(queue, (cost + duration, next_node, path))
        return float("inf"), []


# if __name__ == "__main__":
#     gmaps = GoogleMapsClient()
#
#     engine = create_engine("postgresql+psycopg2://Elena:E!3na2002@localhost:5432/transport")
#     stops_df = pd.read_sql("SELECT stop_id, stop_lat, stop_lon FROM stops", engine)
#     stop_times_df = pd.read_sql("SELECT trip_id, stop_id, stop_sequence FROM stop_times", engine)
#     bikes_df = pd.read_sql("SELECT name, lat, lon FROM shared_vehicle_stations", engine)
#
#     stop_coords = {row.stop_id: (row.stop_lat, row.stop_lon) for _, row in stops_df.iterrows()}
#     bike_coords = {row.name: (row.lat, row.lon) for _, row in bikes_df.iterrows()}
#     bike_stations = list(bike_coords.keys())
#
#     stop_times_df = stop_times_df.sort_values(by=["trip_id", "stop_sequence"]).drop_duplicates()
#     transport_edges = []
#     for trip_id, group in stop_times_df.groupby("trip_id"):
#         stops = list(group.stop_id)
#         transport_edges += list(zip(stops, stops[1:]))
#
#     graph, node_coords = gmaps.build_full_multimodal_graph(transport_edges, bike_stations, stop_coords, bike_coords)
#     gmaps.save_graph_to_json(graph, "graphe_multimodal.json")
#
#     loaded_graph = gmaps.load_graph_from_json("graphe_multimodal.json")
#     print("🌐 Graphe multimodal (extrait) :")
#
#     # Test de recherche de chemin
#     start_node = list(stop_coords.keys())[0]
#     end_node = list(bike_coords.keys())[0]  # Exemple : vers une station vélo
#     total_time, path = gmaps.visualize_shortest_path(graph, node_coords, start_node, end_node)
#     print(f"🧪 Chemin trouvé de '{start_node}' à '{end_node}':")
#     print(" → ".join(path))
#     print(f"⏱️ Durée totale estimée : {total_time:.2f} min")
#     for k, v in list(loaded_graph.items())[:5]:
#         print(k, "=>", v)

if __name__ == "__main__":
    gmaps = GoogleMapsClient()

    engine = create_engine("postgresql+psycopg2://Elena:E!3na2002@localhost:5432/transport")
    stops_df = pd.read_sql("SELECT stop_id, stop_lat, stop_lon FROM stops", engine)
    stop_times_df = pd.read_sql("SELECT trip_id, stop_id, stop_sequence FROM stop_times", engine)
    bikes_df = pd.read_sql("SELECT name, lat, lon FROM shared_vehicle_stations", engine)

    stop_coords = {row.stop_id: (row.stop_lat, row.stop_lon) for _, row in stops_df.iterrows()}
    bike_coords = {row.name: (row.lat, row.lon) for _, row in bikes_df.iterrows()}
    bike_stations = list(bike_coords.keys())

    stop_times_df = stop_times_df.sort_values(by=["trip_id", "stop_sequence"]).drop_duplicates()
    transport_edges = []
    for trip_id, group in stop_times_df.groupby("trip_id"):
        stops = list(group.stop_id)
        transport_edges += list(zip(stops, stops[1:]))

    graph, node_coords = gmaps.build_full_multimodal_graph(transport_edges, bike_stations, stop_coords, bike_coords)

    start_node = list(stop_coords.keys())[0]
    end_node = list(bike_coords.keys())[0]

    gmaps.visualize_shortest_path(graph, node_coords, start_node, end_node)