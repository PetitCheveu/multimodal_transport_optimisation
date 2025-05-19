CREATE EXTENSION postgis;

-- Table des arrêts de transport
CREATE TABLE stops (
    stop_id VARCHAR PRIMARY KEY,
    stop_name TEXT,
    stop_lat DOUBLE PRECISION,
    stop_lon DOUBLE PRECISION,
    location GEOMETRY(Point, 4326)
);

-- Table des trajets
CREATE TABLE trips (
    trip_id VARCHAR PRIMARY KEY,
    route_id VARCHAR,
    route_long_name TEXT
);

-- Table des horaires
CREATE TABLE stop_times (
    id SERIAL PRIMARY KEY,
    trip_id VARCHAR REFERENCES trips(trip_id),
    stop_id VARCHAR REFERENCES stops(stop_id),
    arrival_time TIME,
    departure_time TIME,
    stop_sequence INTEGER,
    pickup_type INTEGER,
    drop_off_type INTEGER
);

-- Table des stations de vélos/trottinettes
CREATE TABLE shared_vehicle_stations (
    station_id VARCHAR PRIMARY KEY,
    name TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    num_bikes INTEGER,
    num_scooters INTEGER,
    location GEOMETRY(Point, 4326)
);
