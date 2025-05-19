import pandas as pd
import psycopg2
import requests
import folium
from psycopg2.extras import execute_values

class GTFSDataExtractor:
    def __init__(self, gtfs_dir="data"):
        """Initialisation avec PostgreSQL et le répertoire GTFS"""
        self.valenciennes_lat, self.valenciennes_lon = 50.357, 3.525
        self.stops_df = None
        self.stop_times_df = None
        self.gtfs_dir = gtfs_dir
        self.conn = psycopg2.connect(dbname="transport", user="Elena", password="E!3na2002", host="localhost", port="5432")
        self.cursor = self.conn.cursor()

        # URLs des données de vélos/trottinettes
        self.station_info_url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_information.json"
        self.station_status_url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_status.json"

    def insert_data(self, table_name, data, columns):
        """Insère les données dans la base PostgreSQL"""
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING;"
        execute_values(self.cursor, query, data)
        self.conn.commit()
        print(f"✔ {len(data)} enregistrements insérés dans {table_name}")

    def find_missing_stop_ids(self):
        """Trouve les stop_id qui existent dans stop_times mais pas dans stops"""
        stop_ids_in_stop_times = set(self.stop_times_df["stop_id"])
        stop_ids_in_stops = set(self.stops_df["stop_id"])
        missing_stops = stop_ids_in_stop_times - stop_ids_in_stops

        if missing_stops:
            print(f"❌ {len(missing_stops)} stop_id manquants dans stops.txt : {missing_stops}")
            return missing_stops
        else:
            print("✅ Tous les stop_id sont présents dans stops.txt.")
            return set()

    def add_missing_parent_stations(self):
        """Ajoute les parent_station manquants pour éviter les erreurs de clé étrangère"""
        print("🔍 Vérification des parent_station manquants...")
        stops_with_parents = self.stops_df[self.stops_df["parent_station"].notna()].copy()

        for _, row in stops_with_parents.iterrows():
            parent_id = str(row["parent_station"])
            if parent_id in self.stops_df["stop_id"].values:
                # Récupérer les infos du parent et assigner les coordonnées
                parent_row = self.stops_df[self.stops_df["stop_id"] == parent_id].iloc[0]
                self.stops_df.loc[self.stops_df["stop_id"] == row["stop_id"], ["stop_lat", "stop_lon"]] = parent_row[
                    ["stop_lat", "stop_lon"]]

        print("✔ Parent_station manquants ajoutés")

    def filter_valid_stop_times(self):
        """Supprime les stop_times dont le stop_id n'existe pas dans stops"""
        valid_stop_times = self.stop_times_df[self.stop_times_df["stop_id"].isin(self.stops_df["stop_id"])]
        print(f"✅ {len(valid_stop_times)} horaires valides sur {len(self.stop_times_df)}")
        return valid_stop_times


    def read_data_and_store_stops(self, datafile):
        """Lit stops.txt et stocke les arrêts de transport dans PostgreSQL"""
        print(f"📥 Chargement des arrêts depuis {datafile}...")
        stops_df = pd.read_csv(datafile)

        # Conversion du type
        stops_df["stop_id"] = stops_df["stop_id"].astype(str)

        stops_data = [
            (row["stop_id"], row["stop_name"], row["stop_lat"], row["stop_lon"], f"POINT({row['stop_lon']} {row['stop_lat']})")
            for _, row in stops_df.iterrows()
        ]
        self.insert_data("stops", stops_data, ["stop_id", "stop_name", "stop_lat", "stop_lon", "location"])

    def read_data_and_store_trips(self, datafile):
        """Lit trips.txt et stocke les trajets dans PostgreSQL"""
        print(f"📥 Chargement des trajets depuis {datafile}...")
        trips_df = pd.read_csv(datafile)

        # Conversion du type
        trips_df["trip_id"] = trips_df["trip_id"].astype(str)

        trips_data = trips_df[["trip_id", "route_id"]].values.tolist()
        self.insert_data("trips", trips_data, ["trip_id", "route_id"])

    def read_data_and_store_stop_times(self, datafile):
        """Lit stop_times.txt et stocke les horaires dans PostgreSQL"""
        print(f"📥 Chargement des horaires depuis {datafile}...")
        stop_times_df = pd.read_csv(datafile)

        # Conversion du type
        stop_times_df["stop_id"] = stop_times_df["stop_id"].astype(str).str.zfill(4)
        stop_times_df["trip_id"] = stop_times_df["trip_id"].astype(str)

        stop_times_data = stop_times_df[["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence", "pickup_type", "drop_off_type"]].values.tolist()
        self.insert_data("stop_times", stop_times_data, ["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence", "pickup_type", "drop_off_type"])

    def fetch_shared_vehicle_data(self):
        """Récupère les zones de vélos et trottinettes avec leur statut"""
        print("🚴 Récupération des stations de vélos et trottinettes...")

        try:
            shared_vehicle_stations = []
            # Récupérer les informations des stations
            station_info_resp = requests.get(self.station_info_url)
            station_status_resp = requests.get(self.station_status_url)

            if station_info_resp.status_code == 200 and station_status_resp.status_code == 200:
                station_info_data = station_info_resp.json()["data"]["stations"]
                station_status_data = {station["station_id"]: station for station in
                                       station_status_resp.json()["data"]["stations"]}

                for station in station_info_data:
                    # print("Détail de la station : ", station)
                    station_id = station["station_id"]
                    name = station["name"]
                    lat = station["lat"]
                    lon = station["lon"]
                    # print("Détail de la station : ", station_status_data.get(station_id, {}))

                    # Vérifier si on a un statut pour cette station
                    # num_bikes = station_status_data.get(station_id, {}).get("num_bikes_available", 0)
                    num_docks = station_status_data.get(station_id, {}).get("num_docks_available", 0)

                    num_bikes = 0
                    num_scooters = 0

                    vehicule_types_available = station_status_data.get(station_id, {}).get("vehicle_types_available", 0)
                    for vehicule_type in vehicule_types_available:
                        if vehicule_type["vehicle_type_id"] == "bike":
                            num_bikes = vehicule_type["count"]
                        elif vehicule_type["vehicle_type_id"] == "scooter":
                            num_scooters = vehicule_type["count"]

                    shared_vehicle_stations.append({
                        "station_id": station_id,
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "num_bikes": num_bikes,
                        "num_scooters": num_scooters,
                        "num_docks": num_docks
                    })

                shared_vehicle_stations = pd.DataFrame(shared_vehicle_stations)

                stations_data = [
                    (row["station_id"], row["name"], row["lat"], row["lon"], row["num_bikes"], row["num_scooters"],
                     f"POINT({row['lon']} {row['lat']})")
                    for _, row in shared_vehicle_stations.iterrows()
                ]
                self.insert_data("shared_vehicle_stations", stations_data,
                                 ["station_id", "name", "lat", "lon", "num_bikes", "num_scooters", "location"])


                print(f"✔ {len(shared_vehicle_stations)} stations de vélos/trottinettes récupérées.")
            else:
                print("❌ Erreur lors de la récupération des données.")
        except Exception as e:
            print(f"❌ Erreur : {e}")

    def read_data_and_store_shared_vehicles(self, datafile):
        """Lit un fichier JSON contenant les stations de vélos/trottinettes et les stocke"""
        print(f"📥 Chargement des stations de vélos/trottinettes depuis {datafile}...")
        shared_vehicles_df = pd.read_json(datafile)

        stations_data = [
            (row["station_id"], row["name"], row["lat"], row["lon"], row["num_bikes"], row["num_scooters"],
             f"POINT({row['lon']} {row['lat']})")
            for _, row in shared_vehicles_df.iterrows()
        ]
        self.insert_data("shared_vehicle_stations", stations_data,
                         ["station_id", "name", "lat", "lon", "num_bikes", "num_scooters", "location"])

    def generate_map(self):
        """Génère une carte des arrêts de transport"""
        print("🗺️ Génération de la carte des arrêts...")

        # Créer une carte centrée sur Valenciennes
        map_valenciennes = folium.Map(location=[self.valenciennes_lat, self.valenciennes_lon], zoom_start=12)

        shared_vehicle_stations = self.cursor.execute("SELECT * FROM shared_vehicle_stations").all()

        # Ajouter les stations de vélos/trottinettes en ROUGE
        for station in shared_vehicle_stations:
            popup_text = f"🚲 {station['name']}<br>Vélos: {station['num_bikes']} | Trottinettes: {station['num_scooters']}<br>Places: {station['num_docks']}"
            folium.Marker(
                [station["lat"], station["lon"]],
                popup=popup_text,
                icon=folium.Icon(color="red", icon="bicycle")
            ).add_to(map_valenciennes)

        # Ajouter uniquement les arrêts valides (physiques)
        for _, stop in self.stops_df[self.stops_df["location"].notna()]:
            folium.Marker(
                [stop["stop_lat"], stop["stop_lon"]],
                popup=f"🚏 {stop['stop_name']}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(map_valenciennes)

        # Sauvegarde de la carte
        map_file = "map.html"
        map_valenciennes.save(map_file)
        print(f"✔ Carte enregistrée dans '{map_file}'.")

    def run(self):
        """Exécute l'extraction et stocke dans PostgreSQL"""
        self.read_data_and_store_stops(f"{self.gtfs_dir}/stops.txt")
        self.fetch_shared_vehicle_data()
        self.generate_map()

        self.read_data_and_store_trips(f"{self.gtfs_dir}/trips.txt")
        self.read_data_and_store_stop_times(f"{self.gtfs_dir}/stop_times.txt")

# Exécuter l'extraction et l'insertion dans PostgreSQL
if __name__ == "__main__":
    gtfs = GTFSDataExtractor()
    gtfs.run()
