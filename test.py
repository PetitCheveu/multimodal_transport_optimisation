import time
from random import randint

from GTFSProcessor import GTFSProcessor
from GoogleApi import GoogleMapsClient
from dotenv import load_dotenv
from logger import setup_logger

if __name__ == "__main__":
    init_time = time.time()
    logger = setup_logger("logs/test_transport_emissions.log")
    load_dotenv()
    gmaps = GoogleMapsClient(logger=logger)
    processor = GTFSProcessor(gtfs_dir="data", logger=logger)
    nb_essais = 100

    # Testing the graph:
    for i in range(nb_essais):
        starting_index = randint(0, 1266)
        ending_index = randint(0, 12660) // 1266
        processor.load_graph_from_json("graph_results/graphe_complet.json")
        logger.info(f"Start: {starting_index}, End: {ending_index}")
        logger.info(f"=== 🧪 Test {i}.1 : With both bicycle and transport allowed duration ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_1.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True, cost_type="duration", visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.2 : With both bicycle and transport allowed distance ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_2.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True, cost_type="distance",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.3 : With both bicycle and transport allowed co2 ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_3.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True, cost_type="co2",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.4 : With both bicycle and transport allowed nb_stops ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_4.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True, cost_type="nb_stops",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.5 : With transport allowed duration ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_5.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=False, allow_transport=True, cost_type="duration",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.6 : With transport allowed distance ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_6.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=False, allow_transport=True, cost_type="distance",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.7 : With transport allowed co2 ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_7.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=False, allow_transport=True, cost_type="co2",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        logger.info(f"=== 🧪 Test {i}.8 : With transport allowed nb_stops ===")
        start = list(processor.node_coords.keys())[starting_index]
        end = list(processor.node_coords.keys())[ending_index]
        logger.info(f"Start: {start}, End: {end}")
        map_filename = f"maps_results/tests/test_all_{i}_8.html"
        start_time = time.time()
        processor.visualize_shortest_path(start, end, allow_bike=False, allow_transport=True, cost_type="nb_stops",
                                          visual_filename=map_filename)
        end_time = time.time()
        logger.info(f"Execution time: {(end_time - start_time):.2f} seconds")

        print(f"Test {i} completed.")
    print("finished in ", time.time() - init_time, " seconds")