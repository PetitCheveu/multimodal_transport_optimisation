import time
from GTFSProcessor import GTFSProcessor
from GoogleApi import GoogleMapsClient
from dotenv import load_dotenv
from logger import setup_logger

if __name__ == "__main__":
    logger = setup_logger("logs/demo_transport_emissions.log")
    print("=== 🧪 Test Démonstration : Autorisation de tous les véhicules ===")
    load_dotenv()
    gmaps = GoogleMapsClient(logger=logger)
    processor = GTFSProcessor(gtfs_dir="data", logger=logger)

    print("=== 🗺️ Chargement des données GTFS ===")
    processor.load_graph_from_json("graph_results/graphe_complet.json")

    start = list(processor.node_coords.keys())[25]
    end = list(processor.node_coords.keys())[138]
    start_time = time.time()
    print("=== 🚶‍♂️ Calcul du chemin le plus court entre deux arrêts ===")
    logger.info("=== 🚶‍♂️ Calcul du chemin le plus court entre deux arrêts ===")
    logger.info(f"Départ: {start}, Arrivée: {end}")
    processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True, visual_filename="maps_results/demo.html")
    end_time = time.time()
    print(f"Temps d'exécution : {(end_time - start_time):.2f} secondes")
    logger.info(f"Temps d'exécution : {(end_time - start_time):.2f} secondes")

