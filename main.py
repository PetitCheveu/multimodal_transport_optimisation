import time
from GTFSProcessor import GTFSProcessor
from GoogleApi import GoogleMapsClient
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    gmaps = GoogleMapsClient()
    processor = GTFSProcessor(gtfs_dir="data")

    # Step 1: Load GTFS data from files
    processor.load_dataframes()

    # Step 2: Insert data into the PostgreSQL database
    processor.insert_into_db()

    # Step 3: Fetch bike-sharing stations
    processor.fetch_shared_vehicle_stations()

    # Step 4: Build the graph from GTFS trips
    processor.build_graph_from_trips()

    # Step 5: Simplify the graph (merge very close stops with the same name)
    processor.simplify_graph(distance_threshold=0.02)  # ≈ 20 meters

    # Step 6: Save the simplified graph
    processor.save_graph_to_json("graph_results/graphe_simplifie.json")

    # Step 7: Enrich the graph with links between bike-sharing stations (Google API)
    processor.enrich_with_bike_stations(gmaps)

    # Step 8: Enrich with estimated walking trips between nearby nodes
    processor.enrich_with_pedestrian_links(max_dist_km=0.2)

    # Step 9: Enrich the graph with emissions data from GTFS route types
    processor.enrich_transport_emissions_from_routes()

    # Step 10: Save the fully enriched graph
    processor.save_graph_to_json("graph_results/graphe_complet.json")


    # Testing the graph:

    processor.load_graph_from_json("graph_results/graphe_complet.json")

    print("=== 🧪 Test 1: With bicycle allowed ===")
    start = list(processor.node_coords.keys())[0]
    end = list(processor.node_coords.keys())[-1]
    start_time = time.time()
    processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=False, visual_filename="maps_results/test_bike.html")
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.2f} seconds")

    print("=== 🧪 Test 2: Walking only ===")
    walk_nodes = list(processor.node_coords.keys())[:2]  # two very close nodes
    start_time = time.time()
    processor.visualize_shortest_path(walk_nodes[0], walk_nodes[1], allow_bike=False, allow_transport=False, visual_filename="maps_results/test_marche.html")
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.2f} seconds")

    print("=== 🧪 Test 3: Public transport only ===")
    transport_nodes = [n for n in processor.graph if any(e['mode'] == 'transport' for e in processor.graph[n])]
    start_time = time.time()
    processor.visualize_shortest_path(transport_nodes[0], transport_nodes[2], allow_bike=False, visual_filename="maps_results/test_transport.html")
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.2f} seconds")

    print("=== 🧪 Test 4: Public transport only ===")
    transport_nodes = [n for n in processor.graph if any(e['mode'] == 'transport' for e in processor.graph[n])]
    start_time = time.time()
    processor.visualize_shortest_path(transport_nodes[0], transport_nodes[1000], allow_bike=False, visual_filename="maps_results/test_transport2.html")
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.2f} seconds")

    print("=== 🧪 Test 5: Random ===")
    transport_nodes = [n for n in processor.graph if any(e['mode'] == 'transport' for e in processor.graph[n])]
    start_time = time.time()
    processor.visualize_shortest_path(transport_nodes[800], transport_nodes[1250], allow_bike=True, visual_filename="maps_results/test_transport3.html")
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.2f} seconds")

    print("=== 🧪 Test 6: With both bicycle and transport allowed ===")
    start = list(processor.node_coords.keys())[0]
    end = list(processor.node_coords.keys())[-1]
    start_time = time.time()
    processor.visualize_shortest_path(start, end, allow_bike=True, allow_transport=True, visual_filename="maps_results/test_bike2.html")
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.2f} seconds")
