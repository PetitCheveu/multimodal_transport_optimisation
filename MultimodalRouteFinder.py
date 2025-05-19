import pandas as pd
import numpy as np
import psycopg2
from geopy.distance import geodesic


class MultimodalRouteFinder:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

    def get_nearest_stop(self, point, radius_km=1):
        query = """
            SELECT stop_id, stop_name, stop_lat, stop_lon,
                   (6371 * acos(
                        cos(radians(%s)) * cos(radians(stop_lat)) *
                        cos(radians(stop_lon) - radians(%s)) +
                        sin(radians(%s)) * sin(radians(stop_lat))
                   )) AS distance
            FROM stops
            WHERE (6371 * acos(
                        cos(radians(%s)) * cos(radians(stop_lat)) *
                        cos(radians(stop_lon) - radians(%s)) +
                        sin(radians(%s)) * sin(radians(stop_lat))
                   )) <= %s
            ORDER BY distance ASC
            LIMIT 1;
        """
        self.cursor.execute(query, (point[0], point[1], point[0], point[0], point[1], point[0], radius_km))
        result = self.cursor.fetchone()
        if result:
            return {
                "stop_id": result[0],
                "stop_name": result[1],
                "stop_lat": result[2],
                "stop_lon": result[3]
            }
        return None

    def find_direct_trip(self, stop_a_id, stop_b_id):
        query = """
            SELECT sta.trip_id, sta.arrival_time AS departure_time, stb.arrival_time AS arrival_time
            FROM stop_times sta
            JOIN stop_times stb ON sta.trip_id = stb.trip_id
            WHERE sta.stop_id = %s AND stb.stop_id = %s AND sta.stop_sequence < stb.stop_sequence
            LIMIT 1;
        """
        self.cursor.execute(query, (stop_a_id, stop_b_id))
        result = self.cursor.fetchone()
        if result:
            trip_id, dep_time, arr_time = result
            self.cursor.execute("SELECT route_id FROM trips WHERE trip_id = %s", (trip_id,))
            route_id = self.cursor.fetchone()[0]
            self.cursor.execute("SELECT route_long_name FROM routes WHERE route_id = %s", (route_id,))
            route_name = self.cursor.fetchone()[0]
            return {
                "trip_id": trip_id,
                "route_name": route_name,
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "from_stop": stop_a_id,
                "to_stop": stop_b_id
            }
        return None

    def find_route_between(self, coord_a, coord_b):
        print("🔍 Recherche du trajet entre les deux points...")
        stop_a = self.get_nearest_stop(coord_a)
        stop_b = self.get_nearest_stop(coord_b)

        if stop_a is None or stop_b is None:
            return {"error": "Aucun arrêt proche trouvé pour A ou B."}

        print(f"🅰️ Stop A: {stop_a['stop_name']} ({stop_a['stop_id']})")
        print(f"🅱️ Stop B: {stop_b['stop_name']} ({stop_b['stop_id']})")

        trip = self.find_direct_trip(stop_a['stop_id'], stop_b['stop_id'])

        if trip:
            print("✅ Trajet direct trouvé !")
            return trip
        else:
            print("⚠️ Aucun trajet direct trouvé pour ces arrêts.")
            return {"error": "Pas de trajet direct disponible pour ces arrêts."}


# Exemple d'utilisation :
if __name__ == "__main__":
    db_params = {
        "dbname": "transport",
        "user": "Elena",
        "password": "E!3na2002",
        "host": "localhost",
        "port": "5432"
    }
    finder = MultimodalRouteFinder(db_config=db_params)
    coord_depart = (50.3566, 3.5225)  # Valenciennes Gare
    coord_arrivee = (50.360721, 3.520527)  # Clémenceau

    trajet = finder.find_route_between(coord_depart, coord_arrivee)
    print(trajet)
