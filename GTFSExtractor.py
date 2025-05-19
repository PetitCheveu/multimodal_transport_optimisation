import pandas as pd
import numpy as np
import folium
import requests


class GTFSExtractor:
    def __init__(self, gtfs_dir="data"):
        """Initialisation avec le répertoire GTFS"""
        self.gtfs_dir = gtfs_dir
        self.stops_file = f"{gtfs_dir}/stops.txt"
        self.stop_times_file = f"{gtfs_dir}/stop_times.txt"
        self.trips_file = f"{gtfs_dir}/trips.txt"
        self.routes_file = f"{gtfs_dir}/routes.txt"
        self.horaires_file = f"{gtfs_dir}/horaires_arrets_valenciennes.csv"
        self.valenciennes_lat, self.valenciennes_lon = 50.357, 3.525

        # URLs des données de vélos/trottinettes
        self.station_info_url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_information.json"
        self.station_status_url = "https://stables.donkey.bike/api/public/gbfs/2/donkey_valenciennes/en/station_status.json"

        # Charger les fichiers GTFS
        self.load_data()
        # Initialiser une liste pour les stations de vélo/trottinettes
        self.shared_vehicle_stations = []

    def load_data(self):
        """Charge les fichiers GTFS nécessaires"""
        print("📥 Chargement des fichiers GTFS...")
        self.stops_df = pd.read_csv(self.stops_file)
        self.stop_times_df = pd.read_csv(self.stop_times_file)
        self.trips_df = pd.read_csv(self.trips_file)
        self.routes_df = pd.read_csv(self.routes_file)

        self.stops_df["stop_lat"] = self.stops_df["stop_lat"].astype(float)
        self.stops_df["stop_lon"] = self.stops_df["stop_lon"].astype(float)

        print(f"✅ {len(self.stops_df)} arrêts chargés")
        print(f"✅ {len(self.stop_times_df)} horaires chargés")

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calcule la distance en km entre deux points géographiques (Haversine)"""
        R = 6371  # Rayon de la Terre en km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def filter_stops_near_valenciennes(self):
        """Filtre les arrêts physiques situés à ≤ 20 km de Valenciennes"""
        print("🔍 Filtrage des arrêts proches de Valenciennes...")

        # Calculer la distance pour chaque arrêt
        self.stops_df["distance_km"] = self.stops_df.apply(
            lambda row: self.haversine(self.valenciennes_lat, self.valenciennes_lon, row["stop_lat"], row["stop_lon"]),
            axis=1
        )

        # Filtrer uniquement les arrêts physiques (`location_type == 0`) et ≤ 20 km
        self.filtered_stops = self.stops_df[
            (self.stops_df["location_type"] == 0) & (self.stops_df["distance_km"] <= 20)
            ].copy()

        print(f"✔ {len(self.filtered_stops)} arrêts sélectionnés.")

    def fetch_shared_vehicle_data(self):
        """Récupère les zones de vélos et trottinettes avec leur statut"""
        print("🚴 Récupération des stations de vélos et trottinettes...")

        try:
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

                    self.shared_vehicle_stations.append({
                        "station_id": station_id,
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "num_bikes": num_bikes,
                        "num_scooters": num_scooters,
                        "num_docks": num_docks
                    })

                print(f"✔ {len(self.shared_vehicle_stations)} stations de vélos/trottinettes récupérées.")
            else:
                print("❌ Erreur lors de la récupération des données.")
        except Exception as e:
            print(f"❌ Erreur : {e}")

    # def extract_schedules(self):
    #     """Extrait les horaires pour les arrêts sélectionnés"""
    #     print("⏳ Extraction des horaires...")
    #
    #     # Associer les stop_id du stop_times.txt avec les stop_id sélectionnés
    #     stop_times_filtered = self.stop_times_df[
    #         self.stop_times_df['stop_id'].isin(self.filtered_stops['stop_id'])
    #     ]
    #
    #     print(f"🎯 {len(stop_times_filtered)} horaires trouvés pour les arrêts sélectionnés.")
    #
    #     # Joindre avec trips.txt pour récupérer route_id
    #     merged_data = stop_times_filtered.merge(self.trips_df[['trip_id', 'route_id']], on='trip_id', how='left')
    #
    #     # Joindre avec routes.txt pour récupérer le nom de la ligne
    #     merged_data = merged_data.merge(self.routes_df[['route_id', 'route_long_name']], on='route_id', how='left')
    #
    #     # Joindre avec stops.txt pour récupérer les infos des arrêts
    #     merged_data = merged_data.merge(
    #         self.filtered_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id', how='left'
    #     )
    #
    #     # Trier les données
    #     merged_data = merged_data.sort_values(by=['stop_id', 'arrival_time'])
    #
    #     # Sauvegarde des données
    #     merged_data.to_csv(self.horaires_file, index=False)
    #     print(f"✔ Horaires enregistrés dans '{self.horaires_file}' avec {len(merged_data)} passages.")

    def extract_schedules(self):
        """Extrait les horaires pour les arrêts sélectionnés"""
        print("⏳ Extraction des horaires...")

        # 🔹 Vérifier la correspondance entre `stop_id` de `stops.txt` et `stop_times.txt`
        stop_ids_filtered = self.filtered_stops["stop_id"].astype(str)
        stop_times_filtered = self.stop_times_df[self.stop_times_df["stop_id"].astype(str).isin(stop_ids_filtered)]

        print(f"🎯 {len(stop_times_filtered)} horaires trouvés pour les arrêts sélectionnés.")

        # 🔹 Joindre avec trips.txt pour récupérer `route_id`
        merged_data = stop_times_filtered.merge(self.trips_df[['trip_id', 'route_id']], on='trip_id', how='left')

        # 🔹 Joindre avec routes.txt pour récupérer `route_long_name`
        merged_data = merged_data.merge(self.routes_df[['route_id', 'route_long_name']], on='route_id', how='left')

        # Forcer stop_id en str dans les deux DataFrames
        merged_data["stop_id"] = merged_data["stop_id"].astype(str)
        self.filtered_stops["stop_id"] = self.filtered_stops["stop_id"].astype(str)

        # 🔹 Joindre avec stops.txt pour récupérer `stop_name`, `stop_lat`, `stop_lon`
        merged_data = merged_data.merge(
            self.filtered_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id', how='left'
        )

        # 🔹 Trier les données
        merged_data = merged_data.sort_values(by=['trip_id', 'stop_sequence'])

        # 🔹 Sélectionner les colonnes demandées
        final_columns = [
            "trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
            "pickup_type", "drop_off_type", "route_id", "route_long_name",
            "stop_name", "stop_lat", "stop_lon"
        ]
        merged_data = merged_data[final_columns]

        # 🔹 Sauvegarde des données
        merged_data.to_csv(self.horaires_file, index=False)
        print(f"✔ Horaires enregistrés dans '{self.horaires_file}' avec {len(merged_data)} passages.")

    def generate_map(self):
        """Génère une carte des arrêts de transport"""
        print("🗺️ Génération de la carte des arrêts...")

        # Créer une carte centrée sur Valenciennes
        map_valenciennes = folium.Map(location=[self.valenciennes_lat, self.valenciennes_lon], zoom_start=12)

        # Ajouter uniquement les arrêts valides (physiques)
        for _, stop in self.filtered_stops.iterrows():
            folium.Marker(
                [stop["stop_lat"], stop["stop_lon"]],
                popup=f"🚏 {stop['stop_name']}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(map_valenciennes)

        # Ajouter les stations de vélos/trottinettes en ROUGE
        for station in self.shared_vehicle_stations:
            popup_text = f"🚲 {station['name']}<br>Vélos: {station['num_bikes']} | Trottinettes: {station['num_scooters']}<br>Places: {station['num_docks']}"
            folium.Marker(
                [station["lat"], station["lon"]],
                popup=popup_text,
                icon=folium.Icon(color="red", icon="bicycle")
            ).add_to(map_valenciennes)

        # Sauvegarde de la carte
        map_file = "maps_results/map.html"
        map_valenciennes.save(map_file)
        print(f"✔ Carte enregistrée dans '{map_file}'.")

    def run(self):
        """Exécute le pipeline complet"""
        self.filter_stops_near_valenciennes()
        self.fetch_shared_vehicle_data()
        self.extract_schedules()
        self.generate_map()


# Lancer l'extraction
if __name__ == "__main__":
    gtfs = GTFSExtractor()
    gtfs.run()
