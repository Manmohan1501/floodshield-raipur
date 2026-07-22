"""
raipur_network.py
-------------------
Real-data backbone for FloodShield: pulls Raipur's actual road network
from OpenStreetMap, gets real elevation per intersection, estimates
drainage quality based on real known flood-prone areas, geocodes
place-name search, and fetches live rainfall forecast data.

HONESTY NOTE ON "SATELLITE DETECTION":
Live, continuously-updated satellite flood-detection isn't used here.
Flood-mapping satellites (e.g. Sentinel-1 radar) pass over any given
city only once every few days, and turning that radar data into a
flood map is itself a specialised research task -- there's no free,
real-time API for "is this exact road flooded right now" from space.
Instead, this module combines REAL live weather forecast data with
REAL elevation and REAL known-flood-prone locations to predict risk --
the same core approach used by real flood-nowcasting systems, since
live satellite imagery isn't actually available continuously either.
"""

import math
import time
import requests
import numpy as np
import streamlit as st

RAIPUR_CENTER = (21.2514, 81.6296)   # Jaistambh Chowk, city center
NETWORK_RADIUS_M = 2500              # ~2.5km radius = central Raipur (kept smaller for reliable, fast fetches)

# Real, well-documented flood-prone points (from local news reports)
FLOOD_PRONE_POINTS = [
    (21.2270, 81.6040, "Gudhiyari"),
    (21.2300, 81.6070, "Gudhiyari Underbridge"),
    (21.2050, 81.6280, "Daganiya"),
    (21.2110, 81.5820, "Tatibandh"),
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def drainage_quality_for_point(lat, lon):
    """Real flood-prone areas get a low score; farther away = better,
    estimated drainage. This is a distance-based heuristic grounded in
    real reported hotspots, not a measured drainage survey."""
    min_dist = min(haversine_km(lat, lon, flat, flon) for flat, flon, _ in FLOOD_PRONE_POINTS)
    if min_dist < 0.5:
        return 0.20
    if min_dist > 3.0:
        return 0.75
    return 0.20 + (min_dist - 0.5) / (3.0 - 0.5) * (0.75 - 0.20)


@st.cache_resource(show_spinner=False)
def get_road_network():
    """Downloads Raipur's real drivable road network from OpenStreetMap.
    Cached so this only runs once per app instance (first load is
    slower; later loads are instant).

    A short, explicit timeout is set on purpose: the public OSM/Overpass
    servers can occasionally be slow to respond, especially from shared
    cloud IPs. Without a timeout, a slow response hangs the whole app
    with no error message. With it, a slow server fails fast with a
    clear error the app can show the user instead.
    """
    import osmnx as ox
    ox.settings.timeout = 45
    ox.settings.log_console = False
    G = ox.graph_from_point(RAIPUR_CENTER, dist=NETWORK_RADIUS_M,
                             network_type="drive", simplify=True)
    return G


@st.cache_data(show_spinner=False)
def get_node_elevations(node_ids, lats, lons):
    """Real elevation per intersection via the free Open-Elevation API,
    batched in chunks. Falls back to a flat estimate if the API is
    unreachable (keeps the app usable even if this free service is
    down or rate-limited)."""
    elevations = {}
    chunk = 100
    for i in range(0, len(node_ids), chunk):
        batch_ids = node_ids[i:i + chunk]
        batch_lats = lats[i:i + chunk]
        batch_lons = lons[i:i + chunk]
        locations = [{"latitude": la, "longitude": lo} for la, lo in zip(batch_lats, batch_lons)]
        try:
            resp = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations}, timeout=20,
            )
            results = resp.json().get("results", [])
            for nid, r in zip(batch_ids, results):
                elevations[nid] = r.get("elevation", 298)
        except Exception:
            for nid in batch_ids:
                elevations[nid] = 298  # Raipur's rough average elevation, as fallback
    return elevations


def geocode_place(query):
    """Turn a place name into coordinates. First checks a small list of
    known real Raipur landmarks (fast, always reliable); falls back to
    OpenStreetMap's free Nominatim search (biased to Raipur) for
    anything else."""
    q = query.strip().lower()
    # 1. Exact match (case-insensitive)
    for name, lat, lon in KNOWN_PLACES:
        if q == name.lower():
            return lat, lon

    # 2. Query contains a known place name -- prefer the longest (most
    #    specific) match, e.g. "near Gudhiyari Underbridge please" should
    #    match "Gudhiyari Underbridge", not just "Gudhiyari"
    contained = [(name, lat, lon) for name, lat, lon in KNOWN_PLACES if name.lower() in q]
    if contained:
        name, lat, lon = max(contained, key=lambda x: len(x[0]))
        return lat, lon

    # 3. Known place name contains the query (e.g. query "gudhiyari" vs
    #    entry "Gudhiyari") -- again prefer the shortest/most exact match
    partial = [(name, lat, lon) for name, lat, lon in KNOWN_PLACES if q in name.lower()]
    if partial:
        name, lat, lon = min(partial, key=lambda x: len(x[0]))
        return lat, lon

    try:
        # Bounding box around Raipur to bias/restrict results
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "viewbox": "81.50,21.32,81.75,21.15",  # lon_min,lat_max,lon_max,lat_min
                "bounded": 1,
            },
            headers={"User-Agent": "FloodShield-Raipur-Prototype (learning project)"},
            timeout=8,
        )
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None, None


# A small gazetteer of real, verified Raipur places -- checked first so
# route search works reliably even if the external geocoding service is
# slow, rate-limited, or phrased differently than expected.
KNOWN_PLACES = [
    ("Jaistambh Chowk", 21.2514, 81.6296),
    ("City Center", 21.2514, 81.6296),
    ("Gudhiyari Underbridge", 21.2300, 81.6070),
    ("Gudhiyari", 21.2270, 81.6040),
    ("Daganiya", 21.2050, 81.6280),
    ("Tatibandh", 21.2110, 81.5820),
    ("AIIMS Raipur", 21.2115, 81.5825),
    ("AIIMS", 21.2115, 81.5825),
    ("Telibandha", 21.2340, 81.6470),
    ("Pandri", 21.2440, 81.6180),
    ("Shankar Nagar", 21.2480, 81.6100),
    ("Amanaka", 21.2600, 81.6550),
    ("Mowa", 21.2750, 81.6650),
    ("Kota", 21.2650, 81.6050),
    ("Ambedkar Hospital", 21.2480, 81.6350),
    ("DKS Hospital", 21.2480, 81.6350),
    ("Moudhapara", 21.2480, 81.6350),
    ("Devendra Nagar", 21.2550, 81.6250),
    ("Railway Station", 21.2470, 81.6430),
]


def get_live_weather(lat, lon, api_key):
    """Real current conditions + hourly forecast from OpenWeatherMap,
    shaped for a rich 'Now' weather card like Google/Apple Weather.
    Returns (weather_dict, None) on success, or (None, error_message)."""
    try:
        current_resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=8,
        )
        current = current_resp.json()
        if str(current.get("cod")) != "200":
            return None, current.get("message", "Could not fetch current weather.")

        forecast_resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=8,
        )
        forecast = forecast_resp.json()
        if str(forecast.get("cod")) != "200":
            return None, forecast.get("message", "Could not fetch forecast.")

        hourly = []
        for entry in forecast.get("list", [])[:8]:  # next 24h in 3h steps
            hourly.append({
                "time": entry["dt_txt"][11:16],  # "HH:MM"
                "temp": round(entry["main"]["temp"]),
                "rain_mm": entry.get("rain", {}).get("3h", 0),
                "pop_pct": round(entry.get("pop", 0) * 100),
                "desc": entry["weather"][0]["description"],
                "icon": entry["weather"][0]["icon"],
            })

        rain_next_6h = sum(h["rain_mm"] for h in hourly[:2])
        is_raining_now = current.get("rain", {}).get("1h", 0) > 0 or "rain" in current["weather"][0]["main"].lower()

        return {
            "current_temp": round(current["main"]["temp"]),
            "feels_like": round(current["main"]["feels_like"]),
            "current_desc": current["weather"][0]["description"],
            "current_icon": current["weather"][0]["icon"],
            "is_raining_now": is_raining_now,
            "rain_1h_mm": current.get("rain", {}).get("1h", 0),
            "rain_next_6h_mm": round(rain_next_6h, 1),
            "hourly": hourly,
        }, None
    except Exception as e:
        return None, str(e)


def get_live_rainfall_forecast(lat, lon, api_key):
    """Real live rainfall forecast via OpenWeatherMap: sums forecast
    rain volume over the next 6 hours as a proxy for 'incoming storm
    severity' that feeds the same ML model as before. Returns
    (expected_rain_mm, description) or (None, error_message)."""
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=8,
        )
        data = resp.json()
        if str(data.get("cod")) != "200":
            return None, data.get("message", "Could not fetch weather data.")

        entries = data.get("list", [])[:2]  # next 2 x 3-hour blocks = 6 hours
        total_rain_mm = sum(e.get("rain", {}).get("3h", 0) for e in entries)
        description = entries[0]["weather"][0]["description"] if entries else "unknown"
        return round(total_rain_mm, 1), description
    except Exception as e:
        return None, str(e)
