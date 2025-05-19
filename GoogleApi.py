import requests
import os
import json
from dotenv import load_dotenv
import heapq
from sklearn.neighbors import BallTree
import numpy as np
from tqdm import tqdm
import folium

class GoogleMapsClient:
    """
    A client to interact with the Google Maps Distance Matrix API,
    and to build and manipulate a multimodal transport graph enriched with real-world travel data.
    """

    def __init__(self):
        """
        Initializes the GoogleMapsClient by loading the API key from the environment.
        Raises:
            ValueError: If the API key is not found in the environment variables.
        """
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("Google Maps API key not found in .env file under GOOGLE_MAPS_API_KEY")
        self.base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    def get_travel_info(self, origin, destination, mode="walking"):
        """
        Retrieves travel distance and duration from Google Maps API between two points.

        Args:
            origin (tuple): Latitude and longitude of the origin.
            destination (tuple): Latitude and longitude of the destination.
            mode (str): Mode of transport (e.g., "walking", "driving", "bicycling").

        Returns:
            dict or None: Dictionary containing distance in km and duration in minutes, or None if unavailable.
        """
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
        """
        Enriches a graph's edges with real-world distances and durations using the Google Maps API.

        Args:
            graph (dict): Graph data structure with node connections.
            node_coords (dict): Dictionary mapping node IDs to (lat, lon) coordinates.
            mode (str): Transport mode for API queries.
            max_edges (int): Maximum number of API calls (limits cost and quota usage).
        """
        print(f"📡 Enriching graph with Google Maps API (mode: {mode})...")
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
                        print(f"⏹️ Limit reached ({max_edges} edges enriched)")
                        return
        print("✅ Graph successfully enriched with API data")

    def build_full_multimodal_graph(self, transport_edges, bike_stations, stop_coords, bike_coords):
        """
        Constructs a multimodal graph combining transport, walking, and biking connections.

        Args:
            transport_edges (list): List of tuples representing public transport connections.
            bike_stations (list): List of bike station IDs (not used directly in this method).
            stop_coords (dict): Coordinates of transport stops.
            bike_coords (dict): Coordinates of bike stations.

        Returns:
            tuple: A tuple (graph, node_coords) where:
                graph (dict): The constructed multimodal graph.
                node_coords (dict): Dictionary of node coordinates.
        """
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
        """
        Saves the graph to a JSON file.

        Args:
            graph (dict): The graph to save.
            filepath (str): The path to the output JSON file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2)
        print(f"📁 Graph saved to '{filepath}'")

    def load_graph_from_json(self, filepath):
        """
        Loads a graph from a JSON file.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: The loaded graph.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        print(f"📂 Graph loaded from '{filepath}'")
        return graph

    def visualize_shortest_path(self, graph, node_coords, start, end):
        """
        Visualizes the shortest path between two nodes using Folium and Dijkstra's algorithm (based on duration).

        Args:
            graph (dict): The multimodal graph.
            node_coords (dict): Dictionary of node coordinates.
            start (str): ID of the start node.
            end (str): ID of the end node.

        Returns:
            tuple: A tuple (total_cost, path) where:
                total_cost (float): Total travel time in minutes.
                path (list): List of node IDs representing the path.
        """
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
                print("📍 Map saved: chemin_folium.html")
                return cost, path
            for edge in graph.get(node, []):
                next_node = edge["to"]
                duration = edge.get("duration_min")
                if duration is not None:
                    heapq.heappush(queue, (cost + duration, next_node, path))
        return float("inf"), []
