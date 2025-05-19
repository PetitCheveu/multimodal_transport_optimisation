import time
from GTFSProcessor import GTFSProcessor
from GoogleApi import GoogleMapsClient
from dotenv import load_dotenv

if __name__ == "__main__":
    print("=== 🧪 Test Démonstration : Autorisation de tous les véhicules ===")
    load_dotenv()
    gmaps = GoogleMapsClient()
    processor = GTFSProcessor(gtfs_dir="data")

    print("=== 🗺️ Chargement des données GTFS ===")
    processor.load_graph_from_json("graphe_complet_v2.json")

    start = list(processor.node_coords.keys())[313]
    end = list(processor.node_coords.keys())[555]
    start_time = time.time()
    print("=== 🚶‍♂️ Calcul du chemin le plus court entre deux arrêts ===")
    processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True,visual_filename="demo.html")
    end_time = time.time()
    print(f"Temps d'exécution : {(end_time - start_time):.2f} secondes")

