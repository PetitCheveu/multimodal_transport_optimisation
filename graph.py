# import psycopg2
# from geopy.distance import geodesic
# import folium
# from collections import defaultdict
# import heapq
#
#
# # === Classe Graph ===
# class Graph:
#     def __init__(self):
#         self.edges = defaultdict(list)  # nœud -> [(voisin, poids, mode)]
#
#     def add_edge(self, from_node, to_node, weight, mode):
#         self.edges[from_node].append((to_node, weight, mode))
#
#
# # === Construction du graphe multimodal ===
# def build_graph_from_db(cursor, max_walk_km=0.5):
#     graph = Graph()
#
#     cursor.execute("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops;")
#     stops = cursor.fetchall()
#     stops_dict = {stop[0]: (stop[2], stop[3]) for stop in stops}
#
#     cursor.execute("SELECT station_id, name, lat, lon FROM shared_vehicle_stations;")
#     stations = cursor.fetchall()
#     stations_dict = {station[0]: (station[2], station[3]) for station in stations}
#     print("Building graph...")
#     # Transport
#     cursor.execute("""
#         SELECT sta.stop_id, stb.stop_id,
#                EXTRACT(EPOCH FROM stb.arrival_time - sta.departure_time) / 60 AS duration
#         FROM stop_times sta
#         JOIN stop_times stb ON sta.trip_id = stb.trip_id
#         WHERE sta.stop_sequence < stb.stop_sequence
#         AND stb.arrival_time > sta.departure_time
#         LIMIT 5000;
#     """)
#     for from_stop, to_stop, duration in cursor.fetchall():
#         if duration and duration < 120:
#             graph.add_edge(from_stop, to_stop, duration, mode='transport')
#     print(f"🚍 Graphe de transport construit avec {len(graph.edges)} arêtes.")
#     # Marche
#     all_points = {**stops_dict, **stations_dict}
#     for id_a, coord_a in all_points.items():
#         for id_b, coord_b in all_points.items():
#             if id_a == id_b:
#                 continue
#             print(f"Calcul de la distance entre {id_a} et {id_b}...")
#             dist = geodesic(coord_a, coord_b).km
#             if dist <= max_walk_km:
#                 duration = (dist / 5) * 60  # min
#                 graph.add_edge(id_a, id_b, duration, mode='walk')
#     print(f"🗺️ Graphe multimodal construit avec {len(graph.edges)} arêtes.")
#     # Vélo / Trottinette
#     for id_a, coord_a in stations_dict.items():
#         for id_b, coord_b in stations_dict.items():
#             if id_a == id_b:
#                 continue
#             dist = geodesic(coord_a, coord_b).km
#             if dist <= 3.0:
#                 duration = (dist / 15) * 60  # vélo
#                 graph.add_edge(id_a, id_b, duration, mode='bike')
#
#     return graph, {**stops_dict, **stations_dict}
#
#
# # === Dijkstra ===
# def dijkstra(graph, start, end):
#     print(f"🔍 Recherche du chemin entre {start} et {end}...")
#     queue = [(0, start, [])]
#     visited = set()
#
#     while queue:
#         print(f"🔄 Traitement du nœud {start}...")
#         cost, node, path = heapq.heappop(queue)
#         if node in visited:
#             continue
#         visited.add(node)
#         path = path + [node]
#         if node == end:
#             return cost, path
#         for neighbor, weight, mode in graph.edges[node]:
#             if neighbor not in visited:
#                 heapq.heappush(queue, (cost + weight, neighbor, path))
#
#     return float("inf"), []
#
#
# # === Affichage du trajet ===
# def display_path(path_ids, node_coords):
#     map_route = folium.Map(location=[50.357, 3.525], zoom_start=13)
#
#     coords = [node_coords[pid] for pid in path_ids if pid in node_coords]
#     folium.PolyLine(coords, color="blue", weight=5).add_to(map_route)
#
#     for pid in path_ids:
#         if pid in node_coords:
#             lat, lon = node_coords[pid]
#             folium.Marker([lat, lon], popup=f"{pid}").add_to(map_route)
#
#     map_route.save("chemin.html")
#     print("✔ Carte générée : chemin.html")
#
#
# # === Main ===
# if __name__ == "__main__":
#     print("🔍 Chargement du graphe multimodal depuis la base de données...")
#     db_params = {
#         "dbname": "transport",
#         "user": "Elena",
#         "password": "E!3na2002",
#         "host": "localhost",
#         "port": "5432"
#     }
#
#     with psycopg2.connect(**db_params) as conn:
#         # Connexion à la base de données
#         print("🔗 Connexion à la base de données réussie.")
#         cursor = conn.cursor()
#         graph, coords = build_graph_from_db(cursor)
#
#         # 🧪 Exemple de test : remplacer par des vrais ID d’arrêt ou station de ta BDD
#         depart = "0001"   # Exemple d'ID arrêt ou station
#         arrivee = "0015"  # Exemple d'ID arrêt ou station
#
#         print("🔍 Recherche du chemin entre les points A et B...")
#         cost, path = dijkstra(graph, depart, arrivee)
#
#         if path:
#             print(f"🟢 Coût total : {cost:.1f} minutes")
#             print("🛣️ Chemin :", " → ".join(path))
#             display_path(path, coords)
#         else:
#             print("❌ Aucun chemin trouvé.")
import os

import psycopg2
from geopy.distance import geodesic
import folium
from collections import defaultdict
import heapq
import numpy as np
from sklearn.neighbors import BallTree


# === Classe Graph ===
class Graph:
    def __init__(self):
        self.edges = defaultdict(list)

    def add_edge(self, from_node, to_node, weight, mode):
        self.edges[from_node].append((to_node, weight, mode))


# === Récupération des points (stops + stations) ===
def load_points_from_db(cursor):
    cursor.execute("SELECT stop_id, stop_lat, stop_lon FROM stops;")
    stops = cursor.fetchall()

    cursor.execute("SELECT station_id, lat, lon FROM shared_vehicle_stations;")
    stations = cursor.fetchall()

    all_points = {}
    for pid, lat, lon in stops:
        all_points[pid] = (lat, lon)
    for sid, lat, lon in stations:
        all_points[sid] = (lat, lon)

    return all_points


# === Création rapide des arêtes à pied ou en vélo avec BallTree ===
def add_edges_with_balltree(graph, points_dict, max_dist_km, mode="walk", speed_kmh=5):
    ids = list(points_dict.keys())
    coords = np.array([[lat, lon] for lat, lon in points_dict.values()])
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')

    for i, pid in enumerate(ids):
        dists_rad, idxs = tree.query_radius([coords_rad[i]], r=max_dist_km / 6371, return_distance=True)
        for dist, j in zip(dists_rad[0], idxs[0]):
            if i != j:
                duration = (dist * 6371 / speed_kmh) * 60  # min
                graph.add_edge(pid, ids[int(j)], duration, mode=mode)


# === Ajout des trajets transport en commun depuis la base ===
def add_gtfs_edges(graph, cursor):
    cursor.execute("""
        SELECT sta.stop_id, stb.stop_id,
               EXTRACT(EPOCH FROM stb.arrival_time - sta.departure_time) / 60 AS duration
        FROM stop_times sta
        JOIN stop_times stb ON sta.trip_id = stb.trip_id
        WHERE sta.stop_sequence < stb.stop_sequence
        AND stb.arrival_time > sta.departure_time
        LIMIT 5000;
    """)
    for from_stop, to_stop, duration in cursor.fetchall():
        if duration and duration < 120:
            graph.add_edge(from_stop, to_stop, duration, mode='transport')


# === Dijkstra ===
def dijkstra(graph, start, end):
    queue = [(0, start, [])]
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return cost, path
        for neighbor, weight, mode in graph.edges[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path))

    return float("inf"), []


# === Affichage du trajet sur carte ===
def display_path(path_ids, node_coords):
    map_route = folium.Map(location=[50.357, 3.525], zoom_start=13)

    coords = [node_coords[pid] for pid in path_ids if pid in node_coords]
    folium.PolyLine(coords, color="blue", weight=5).add_to(map_route)

    for pid in path_ids:
        if pid in node_coords:
            lat, lon = node_coords[pid]
            folium.Marker([lat, lon], popup=f"{pid}").add_to(map_route)

    map_route.save("chemin.html")
    print("✔ Carte générée : chemin.html")


# === Main ===
if __name__ == "__main__":
    db_params = {
        "dbname": os.getenv("DATABASE_NAME"),
        "user": os.getenv("DATABASE_USER"),
        "password": os.getenv("DATABASE_PASSWORD"),
        "host": os.getenv("DATABASE_HOST"),
        "port": int(os.getenv("DATABASE_PORT", 5432))
    }

    with psycopg2.connect(**db_params) as conn:
        cursor = conn.cursor()

        print("📥 Chargement des points...")
        points = load_points_from_db(cursor)


        print("🛠️ Construction du graphe...")
        graph = Graph()
        add_gtfs_edges(graph, cursor)
        add_edges_with_balltree(graph, points, max_dist_km=0.5, mode="walk", speed_kmh=5)
        add_edges_with_balltree(graph, points, max_dist_km=3.0, mode="bike", speed_kmh=15)



        # 🧪 Exemple : à adapter avec tes identifiants d'arrêt ou station
        depart = "0001"   # ID réel d'un arrêt
        arrivee = "0015"  # ID réel d'un autre arrêt

        print("🔍 Calcul du plus court chemin...")
        cost, path = dijkstra(graph, depart, arrivee)
        print("Départ présent :", depart in points)
        print("Arrivée présente :", arrivee in points)

        print("Nombre de sommets dans le graphe :", len(graph.edges))
        print("Nombre de voisins du départ :", len(graph.edges.get(depart, [])))
        print("Nombre de voisins de l'arrivée :", len(graph.edges.get(arrivee, [])))

        if path:
            print(f"🟢 Coût total : {cost:.1f} minutes")
            print("🛣️ Chemin :", " → ".join(path))

            # ✅ Calcul de la distance géographique entre départ et arrivée
            coord_depart = points[depart]
            coord_arrivee = points[arrivee]
            distance_km = geodesic(coord_depart, coord_arrivee).km
            print(f"📏 Distance entre {depart} et {arrivee} : {distance_km:.2f} km")

            display_path(path, points)
        else:
            print("❌ Aucun chemin trouvé.")

