"""
raipur_data.py
----------------
Real-world-grounded location data for Raipur, Chhattisgarh, used by the
FloodShield prototype.

IMPORTANT / HONESTY NOTE:
Place names, the river, and the flood-prone neighbourhoods below are REAL
and based on public news reports and government sources (see project
README for sources). However, the exact latitude/longitude coordinates,
elevation values, and drainage-quality scores are APPROXIMATE ESTIMATES
for demonstration purposes only -- they are not official surveyed GIS
data. A production version of FloodShield would replace these with real
elevation data (e.g. from a government GIS/DEM source) and real road
network data (e.g. OpenStreetMap).
"""

import pandas as pd

# ---------------------------------------------------------------------
# Intersections / key locations in Raipur
# elevation_m is a rough relative estimate: lower = closer to the Kharun
# river / historically flood-affected low-lying areas, higher = safer
# high ground. Raipur district sits between ~244-409m above sea level.
# ---------------------------------------------------------------------
NODES = [
    # node_id, name,                         lat,      lon,      elevation_m, is_landmark
    dict(node_id=0,  name="Jaistambh Chowk (City Center)", lat=21.2514, lon=81.6296, elevation_m=298, type="landmark"),
    dict(node_id=1,  name="Gudhiyari",                      lat=21.2270, lon=81.6040, elevation_m=268, type="flood_prone"),
    dict(node_id=2,  name="Gudhiyari Underbridge (Kharun bank)", lat=21.2300, lon=81.6070, elevation_m=259, type="flood_prone"),
    dict(node_id=3,  name="Daganiya",                       lat=21.2050, lon=81.6280, elevation_m=271, type="flood_prone"),
    dict(node_id=4,  name="Tatibandh",                      lat=21.2110, lon=81.5820, elevation_m=284, type="flood_prone"),
    dict(node_id=5,  name="AIIMS Raipur (Tatibandh)",       lat=21.2115, lon=81.5825, elevation_m=285, type="shelter"),
    dict(node_id=6,  name="Telibandha Lake area",           lat=21.2340, lon=81.6470, elevation_m=301, type="normal"),
    dict(node_id=7,  name="Pandri Bus Stand",                lat=21.2440, lon=81.6180, elevation_m=295, type="normal"),
    dict(node_id=8,  name="Shankar Nagar",                  lat=21.2480, lon=81.6100, elevation_m=297, type="normal"),
    dict(node_id=9,  name="Amanaka",                        lat=21.2600, lon=81.6550, elevation_m=303, type="normal"),
    dict(node_id=10, name="Mowa",                           lat=21.2750, lon=81.6650, elevation_m=305, type="normal"),
    dict(node_id=11, name="Kota",                           lat=21.2650, lon=81.6050, elevation_m=294, type="normal"),
    dict(node_id=12, name="Dr. B.R. Ambedkar Hospital (DKS), Moudhapara", lat=21.2480, lon=81.6350, elevation_m=296, type="shelter"),
    dict(node_id=13, name="Devendra Nagar",                 lat=21.2550, lon=81.6250, elevation_m=300, type="normal"),
    dict(node_id=14, name="Raipur Railway Station",         lat=21.2470, lon=81.6430, elevation_m=297, type="normal"),
]
city_nodes = pd.DataFrame(NODES)

# ---------------------------------------------------------------------
# Roads connecting the locations above.
# drainage_quality: 0 (poor) - 1 (excellent). Gudhiyari, the Gudhiyari
# Underbridge, and Daganiya are given LOW drainage scores because they
# are repeatedly reported in local news as waterlogging hotspots every
# monsoon. This reflects real reporting, not a precise measurement.
# ---------------------------------------------------------------------
ROADS = [
    dict(road_id=0,  node_a=0,  node_b=8,  drainage_quality=0.55),
    dict(road_id=1,  node_a=0,  node_b=7,  drainage_quality=0.60),
    dict(road_id=2,  node_a=0,  node_b=13, drainage_quality=0.65),
    dict(road_id=3,  node_a=8,  node_b=1,  drainage_quality=0.30),
    dict(road_id=4,  node_a=1,  node_b=2,  drainage_quality=0.15),  # Gudhiyari -> Underbridge, worst
    dict(road_id=5,  node_a=2,  node_b=3,  drainage_quality=0.20),  # Underbridge -> Daganiya
    dict(road_id=6,  node_a=3,  node_b=4,  drainage_quality=0.35),  # Daganiya -> Tatibandh
    dict(road_id=7,  node_a=4,  node_b=5,  drainage_quality=0.50),  # Tatibandh -> AIIMS
    dict(road_id=8,  node_a=1,  node_b=3,  drainage_quality=0.25),
    dict(road_id=9,  node_a=8,  node_b=0,  drainage_quality=0.55),
    dict(road_id=10, node_a=7,  node_b=14, drainage_quality=0.60),
    dict(road_id=11, node_a=14, node_b=6,  drainage_quality=0.65),
    dict(road_id=12, node_a=6,  node_b=9,  drainage_quality=0.70),
    dict(road_id=13, node_a=9,  node_b=10, drainage_quality=0.72),
    dict(road_id=14, node_a=0,  node_b=12, drainage_quality=0.58),
    dict(road_id=15, node_a=12, node_b=13, drainage_quality=0.62),
    dict(road_id=16, node_a=13, node_b=9,  drainage_quality=0.68),
    dict(road_id=17, node_a=0,  node_b=11, drainage_quality=0.60),
    dict(road_id=18, node_a=11, node_b=9,  drainage_quality=0.66),
    dict(road_id=19, node_a=7,  node_b=8,  drainage_quality=0.58),
    dict(road_id=20, node_a=3,  node_b=14, drainage_quality=0.40),
]
roads_df = pd.DataFrame(ROADS)

# Attach elevation info to each road (average of its two endpoints)
elev_lookup = city_nodes.set_index("node_id")["elevation_m"]
roads_df["avg_elevation_m"] = roads_df.apply(
    lambda r: (elev_lookup[r.node_a] + elev_lookup[r.node_b]) / 2, axis=1
)

# ---------------------------------------------------------------------
# Shelters (real hospitals -- can also serve as emergency shelter points)
# ---------------------------------------------------------------------
SHELTERS = [
    dict(name="AIIMS Raipur", node_id=5, address="Tatibandh, G.E. Road, Raipur - 492099",
         phone="0771-2572240", capacity_estimate="Large (tertiary govt. hospital)"),
    dict(name="Dr. B.R. Ambedkar Memorial Hospital (DKS)", node_id=12, address="Jail Road, Moudhapara, Raipur - 492001",
         phone="Contact via Raipur district helpline", capacity_estimate="Large (govt. super-speciality hospital)"),
]
shelters_df = pd.DataFrame(SHELTERS)

# ---------------------------------------------------------------------
# Known historically flood/waterlogging-prone areas (from 2026 local
# news reports -- see README for sources)
# ---------------------------------------------------------------------
HISTORICAL_FLOOD_AREAS = [
    "Gudhiyari Underbridge (Narmada Park underbridge) -- floods almost every monsoon",
    "Daganiya -- repeated severe waterlogging",
    "Tatibandh -- waterlogging during heavy rain events",
]

# ---------------------------------------------------------------------
# Emergency contacts (real, publicly published national/state numbers)
# ---------------------------------------------------------------------
EMERGENCY_CONTACTS = [
    dict(service="All-in-one Emergency (Police/Fire/Medical)", number="112"),
    dict(service="Police Control Room", number="100"),
    dict(service="Fire Helpline", number="101"),
    dict(service="Ambulance", number="108"),
    dict(service="National Disaster Management Helpline", number="1078"),
    dict(service="Women's Helpline", number="1091"),
]

RIVERS = [
    dict(name="Kharun River", note="Tributary of the Shivnath River; passes along the western side of Raipur city near Gudhiyari."),
    dict(name="Mahanadi River", note="Major river of Chhattisgarh, flows through Raipur district."),
]
