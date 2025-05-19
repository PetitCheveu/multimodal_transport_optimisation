import time

from IPython.utils.timing import clock

from GTFSProcessor import GTFSProcessor  # adapte ce nom si ton fichier s'appelle différemment
from GoogleApi import GoogleMapsClient  # ton client API Google
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # processor = GTFSProcessor(gtfs_dir="data")
    # processor.load_dataframes()

    # processor.build_graph_from_trips()
    # print("Nombre de sommets (noeuds) :", len(processor.graph))
    # print(f"Graph : {processor.graph}")
    # print("Exemple d’arêtes sortantes pour la gare :", processor.graph.get("1225", []))  # ou un ID spécifique

    load_dotenv()  # charge les variables d'environnement
    gmaps = GoogleMapsClient()
    processor = GTFSProcessor(gtfs_dir="data")

    # # Étape 1 : Charger les données GTFS depuis les fichiers
    # processor.load_dataframes()
    #
    # # Étape 2 : Insérer les données dans la base PostgreSQL
    # processor.insert_into_db()
    #
    # # Étape 3 : Récupérer les stations de vélo
    # processor.fetch_shared_vehicle_stations()
    #
    # # Étape 4 : Construire le graphe à partir des trajets GTFS
    # processor.build_graph_from_trips()
    #
    # # Étape 5 : Simplifier le graphe (fusion des arrêts très proches avec le même nom)
    # processor.simplify_graph(distance_threshold=0.02)  # ≈ 20 mètres
    #
    # # Étape 6 : Sauvegarder le graphe simplifié
    # processor.save_graph_to_json("graphe_simplifie.json")
    #
    # # Étape 7 : Enrichir avec les liaisons entre stations de vélo (Google API)
    # processor.enrich_with_bike_stations(gmaps)
    #
    # # Étape 8 : Enrichir avec les trajets à pied estimés entre sommets proches
    # processor.enrich_with_pedestrian_links(max_dist_km=0.2)
    #
    # # Étape 9 : Enrichir le graphe avec les données de l'API Google Maps
    # processor.enrich_transport_emissions_from_routes()
    #
    # # Étape 10 : Réenregistrer le graphe enrichi complet
    # processor.save_graph_to_json("graphe_complet.json")
    #
    # processor.load_graph_from_json("graphe_complet.json")

    # Étape finale : Visualiser un trajet entre deux points du graphe
    # start = list(processor.node_coords.keys())[1250]
    # end = list(processor.node_coords.keys())[992]
    # for i in range(len(processor.node_coords)):
    #     print(f"Node {i}: {list(processor.node_coords.keys())[i]}")
    # processor.visualize_shortest_path(start, end, False)

    # coord_start = (50.36053582035317, 3.5208171690536227)
    # coord_end = (50.3246115, 3.5146953)
    #
    # processor.find_and_visualize_trip_from_coordinates(coord_start, coord_end, allow_bike=True)

    processor.load_graph_from_json("graphe_complet.json")  # ou graphe simplifié selon l'étape

    print("=== 🧪 Test 1 : Avec vélo autorisé ===")
    start = list(processor.node_coords.keys())[0]
    end = list(processor.node_coords.keys())[-1]
    start_time = time.time()
    processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=False,visual_filename="test_bike.html")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution : {elapsed_time:.2f} secondes")

    print("=== 🧪 Test 2 : Marche uniquement ===")
    walk_nodes = list(processor.node_coords.keys())[:2]  # deux nœuds très proches
    start_time = time.time()
    processor.visualize_shortest_path(walk_nodes[0], walk_nodes[1], allow_bike=False, allow_transport=False, visual_filename="test_marche.html")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution : {elapsed_time:.2f} secondes")

    print("=== 🧪 Test 3 : Transports en commun uniquement ===")
    transport_nodes = [n for n in processor.graph if any(e['mode'] == 'transport' for e in processor.graph[n])]
    # for i in range(len(transport_nodes)):
    #     print(f"Node {i}: {transport_nodes[i]}")
    start_time = time.time()
    processor.visualize_shortest_path(transport_nodes[0], transport_nodes[2], allow_bike=False, visual_filename="test_transport.html")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution : {elapsed_time:.2f} secondes")

    print("=== 🧪 Test 4 : Transports en commun uniquement ===")
    transport_nodes = [n for n in processor.graph if any(e['mode'] == 'transport' for e in processor.graph[n])]
    # for i in range(len(transport_nodes)):
    #     print(f"Node {i}: {transport_nodes[i]}")
    start_time = time.time()
    processor.visualize_shortest_path(transport_nodes[0], transport_nodes[1000], allow_bike=False,
                                      visual_filename="test_transport2.html")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution : {elapsed_time:.2f} secondes")

    print("=== 🧪 Test 5 : Random===")
    transport_nodes = [n for n in processor.graph if any(e['mode'] == 'transport' for e in processor.graph[n])]
    # for i in range(len(transport_nodes)):
    #     print(f"Node {i}: {transport_nodes[i]}")
    start_time = time.time()
    processor.visualize_shortest_path(transport_nodes[800], transport_nodes[1250], allow_bike=True,
                                      visual_filename="test_transport3.html")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution : {elapsed_time:.2f} secondes")

    # print("=== 🧪 Test 6 : Avec vélo et transport autorisés ===")
    # start = list(processor.node_coords.keys())[0]
    # end = list(processor.node_coords.keys())[-1]
    # # Calcul du temps d'exécution :
    #
    # start_time = time.time()
    # processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True,visual_filename="test_bike2.html")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Temps d'exécution : {elapsed_time:.2f} secondes")