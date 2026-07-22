"""
FloodShield -- Raipur Prototype (v2: real road network + live weather)
------------------------------------------------------------------------
See raipur_network.py for the honesty note on why this doesn't use
live satellite flood-image classification, and what real data it does
use instead (live weather forecast + real elevation + real road
network + real known flood-prone areas).
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
import joblib
import os

import raipur_network as rn
from raipur_data import shelters_df, HISTORICAL_FLOOD_AREAS, EMERGENCY_CONTACTS, RIVERS

st.set_page_config(page_title="FloodShield", page_icon="\U0001F30A", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_model.joblib")
SUPPORTED_CITIES = {"raipur"}


@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    import train_model  # noqa: F401
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat, lon):
    import requests
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 10},
            headers={"User-Agent": "FloodShield-Prototype/1.0"},
            timeout=6,
        )
        addr = resp.json().get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("county") or addr.get("state_district")
        state = addr.get("state")
        return city, state
    except Exception:
        return None, None


@st.cache_resource(show_spinner="Loading Raipur's real road network from OpenStreetMap (first load only, ~30-60s)...")
def load_network_with_features():
    G = rn.get_road_network()
    node_ids = list(G.nodes)
    lats = [G.nodes[n]["y"] for n in node_ids]
    lons = [G.nodes[n]["x"] for n in node_ids]
    elevations = rn.get_node_elevations(tuple(node_ids), tuple(lats), tuple(lons))
    drainage = {n: rn.drainage_quality_for_point(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_ids}
    return G, elevations, drainage


def predict_edge_risk(G, elevations, drainage, rainfall_mm, model):
    edge_keys = list(G.edges(keys=True))
    rows = []
    for u, v, k in edge_keys:
        elev = (elevations.get(u, 298) + elevations.get(v, 298)) / 2
        drain = (drainage.get(u, 0.5) + drainage.get(v, 0.5)) / 2
        rows.append((elev, drain))
    feat = pd.DataFrame(rows, columns=["avg_elevation_m", "drainage_quality"])
    feat["rainfall_mm"] = rainfall_mm
    feat = feat[["rainfall_mm", "avg_elevation_m", "drainage_quality"]]
    probs = model.predict_proba(feat)[:, 1]
    preds = (probs > 0.5).astype(int)
    for (u, v, k), p, pr in zip(edge_keys, probs, preds):
        G.edges[u, v, k]["flood_risk_prob"] = float(p)
        G.edges[u, v, k]["predicted_flooded"] = int(pr)
    return G


def build_routing_graph(G, block_threshold=0.85, risk_penalty=25):
    H = nx.MultiDiGraph()
    H.graph.update(G.graph)
    for n, data in G.nodes(data=True):
        H.add_node(n, **data)
    for u, v, k, data in G.edges(keys=True, data=True):
        risk = data.get("flood_risk_prob", 0)
        if risk > block_threshold:
            continue
        length_km = data.get("length", 100) / 1000
        weight = length_km * (1 + risk * risk_penalty)
        new_data = dict(data)
        new_data["weight"] = weight
        H.add_edge(u, v, key=k, **new_data)
    return H


def route_distance_km(H, route):
    return sum(H.edges[route[i], route[i + 1], 0].get("length", 0)
               for i in range(len(route) - 1) if H.has_edge(route[i], route[i + 1])) / 1000


def get_route_options(G, start_node, end_node):
    """Returns up to 2 route options, similar to Google Maps offering
    alternatives: a 'Safest' route (heavily avoids flood risk, even if
    longer) and a 'Balanced' route (shorter, moderate risk avoidance).
    Each entry: {label, route, latlon, distance_km, avg_risk}."""
    options = []
    strategies = [("Safest (avoids flood risk)", 25, 0.85), ("Balanced (shorter, some risk avoidance)", 5, 0.85)]
    seen_routes = set()
    for label, penalty, threshold in strategies:
        H = build_routing_graph(G, block_threshold=threshold, risk_penalty=penalty)
        if start_node not in H or end_node not in H:
            continue
        try:
            route = nx.shortest_path(H, start_node, end_node, weight="weight")
        except nx.NetworkXNoPath:
            continue
        key = tuple(route)
        if key in seen_routes:
            continue
        seen_routes.add(key)
        risks = [H.edges[route[i], route[i + 1], 0].get("flood_risk_prob", 0)
                 for i in range(len(route) - 1) if H.has_edge(route[i], route[i + 1])]
        options.append({
            "label": label,
            "route": route,
            "latlon": route_to_latlon(H, route),
            "distance_km": route_distance_km(H, route),
            "avg_risk": (sum(risks) / len(risks)) if risks else 0,
        })
    return options


def route_to_latlon(G, route):
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]


def draw_network_map(G, center, route_latlon=None, alt_routes_latlon=None, start_latlon=None,
                      end_latlon=None, user_latlon=None, rainfall_mm=0, max_edges_drawn=2500):
    # Real satellite imagery, like Google Maps satellite view
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", control=False,
    ).add_to(m)

    # Shaded "floodwater" zones over real known flood-prone areas -- radius
    # and opacity scale with how much rain is expected, so it visibly
    # grows during heavier storms.
    zone_radius = 150 + min(rainfall_mm, 150) * 4  # meters
    zone_opacity = min(0.15 + rainfall_mm / 300, 0.55)
    for flat, flon, fname in rn.FLOOD_PRONE_POINTS:
        folium.Circle(
            location=(flat, flon), radius=zone_radius,
            color="#1d4ed8", weight=1, fill=True, fill_color="#3b82f6",
            fill_opacity=zone_opacity, opacity=0.4,
            tooltip=f"Flood-prone area: {fname}",
        ).add_to(m)

    edges = list(G.edges(keys=True, data=True))
    if len(edges) > max_edges_drawn:
        edges = edges[:max_edges_drawn]
    for u, v, k, data in edges:
        risk = data.get("flood_risk_prob", 0)
        flooded = data.get("predicted_flooded", 0)
        lat1, lon1 = G.nodes[u]["y"], G.nodes[u]["x"]
        lat2, lon2 = G.nodes[v]["y"], G.nodes[v]["x"]
        if flooded == 1:
            # "Water hazard" blue-red gradient: certainly-flooded roads
            # rendered in deep water-blue, risky roads in amber
            color = "#0ea5e9" if risk > 0.85 else "#f59e0b"
            weight = 4
        else:
            color = "#e5e7eb"
            weight = 1.5
        folium.PolyLine([[lat1, lon1], [lat2, lon2]], color=color, weight=weight, opacity=0.75).add_to(m)

    # Alternate (non-chosen) route options, shown lighter/dashed
    if alt_routes_latlon:
        for alt in alt_routes_latlon:
            folium.PolyLine(alt, color="#94a3b8", weight=4, opacity=0.6, dash_array="6,8").add_to(m)

    if route_latlon:
        folium.PolyLine(route_latlon, color="limegreen", weight=6, opacity=0.95,
                         tooltip="Recommended route").add_to(m)
    if start_latlon:
        folium.Marker(start_latlon, tooltip="Start", icon=folium.Icon(color="blue", icon="play")).add_to(m)
    if end_latlon:
        folium.Marker(end_latlon, tooltip="Destination", icon=folium.Icon(color="black", icon="flag")).add_to(m)
    if user_latlon:
        folium.Marker(user_latlon, tooltip="Your live location",
                       icon=folium.Icon(color="purple", icon="user")).add_to(m)
    return m


# =======================================================================
# SIDEBAR: real location detection + live weather
# =======================================================================
st.sidebar.title("\U0001F30A FloodShield")
location = get_geolocation()

detected_city, detected_state = (None, None)
user_lat, user_lon = None, None
if location and isinstance(location, dict) and "coords" in location:
    user_lat = location["coords"]["latitude"]
    user_lon = location["coords"]["longitude"]
    detected_city, detected_state = reverse_geocode(user_lat, user_lon)

if detected_city and detected_city.strip().lower() in SUPPORTED_CITIES:
    st.sidebar.success(f"\U0001F4CD Detected location: **{detected_city}, {detected_state}**")
elif detected_city:
    st.sidebar.warning(f"\U0001F4CD Detected location: **{detected_city}, {detected_state}**")
    st.sidebar.info("FloodShield isn't live for your city yet -- showing the **Raipur** demo below.")
else:
    st.sidebar.info("\U0001F4CD Waiting for location permission... Showing the **Raipur** demo below.")

st.sidebar.divider()
st.sidebar.subheader("\U0001F327\uFE0F Live Weather")
owner_key = st.secrets.get("OPENWEATHER_API_KEY") if hasattr(st, "secrets") else None
if owner_key:
    api_key = owner_key
else:
    api_key = st.sidebar.text_input("OpenWeatherMap API key", type="password",
                                     help="Free key from openweathermap.org/api_keys")
    st.sidebar.caption(
        "\u2139\uFE0F The app owner hasn't set up a shared weather key yet, so each "
        "visitor needs to enter their own for live data."
    )

weather_lat, weather_lon = (user_lat, user_lon) if (user_lat and detected_city and
                                                      detected_city.strip().lower() in SUPPORTED_CITIES) else rn.RAIPUR_CENTER

weather, weather_error = (None, None)
if api_key:
    with st.sidebar:
        with st.spinner("Fetching live weather..."):
            weather, weather_error = rn.get_live_weather(weather_lat, weather_lon, api_key)
    if weather:
        icon_url = f"https://openweathermap.org/img/wn/{weather['current_icon']}@2x.png"
        wcol1, wcol2 = st.sidebar.columns([1, 2])
        wcol1.image(icon_url, width=60)
        wcol2.metric("Now", f"{weather['current_temp']}\u00B0C", weather['current_desc'].title())
        if weather["is_raining_now"]:
            st.sidebar.warning(f"\U0001F327\uFE0F Raining now ({weather['rain_1h_mm']} mm in the last hour)")
        st.sidebar.caption(f"Expected over next 6h: **{weather['rain_next_6h_mm']} mm**")

        with st.sidebar.expander("Hourly forecast"):
            for h in weather["hourly"][:6]:
                hicon = f"https://openweathermap.org/img/wn/{h['icon']}.png"
                hc1, hc2, hc3 = st.columns([1, 1, 2])
                hc1.write(h["time"])
                hc2.write(f"{h['temp']}\u00B0")
                hc3.write(f"\U0001F4A7 {h['pop_pct']}% ({h['rain_mm']}mm)")
    else:
        st.sidebar.warning(f"Couldn't fetch live weather yet ({weather_error}). "
                            f"New keys can take up to an hour to activate.")
else:
    st.sidebar.caption("Enter your free OpenWeatherMap key above for live weather. "
                        "Using a manual rainfall value below until then.")

live_rain_mm = weather["rain_next_6h_mm"] if weather else None

st.sidebar.divider()
st.sidebar.subheader("\U0001F6A8 Emergency Contacts")
for c in EMERGENCY_CONTACTS:
    st.sidebar.write(f"**{c['service']}**: {c['number']}")

st.sidebar.divider()
st.sidebar.caption(
    "\u26A0\uFE0F This prototype does not use live satellite flood-image "
    "detection (not realistically available in real time). It combines "
    "live weather forecast data with real elevation and real known "
    "flood-prone areas instead. See README for details and sources."
)

# =======================================================================
# MAIN: load the real network (once, cached)
# =======================================================================
st.title("\U0001F30A FloodShield -- Raipur")

try:
    G, elevations, drainage = load_network_with_features()
    network_ok = True
except Exception as e:
    network_ok = False
    st.error(
        "Couldn't load the real road network right now -- OpenStreetMap's public "
        "servers may be busy or slow to respond. This is usually temporary. "
        f"Technical detail: {e}"
    )
    if st.button("Retry"):
        st.rerun()

if network_ok:
    model = load_model()

    rainfall_mm = live_rain_mm if live_rain_mm is not None else st.slider(
        "Manual rainfall input (mm, next 6h) -- used until a live weather key is added",
        0, 150, 40, 5,
    )

    G = predict_edge_risk(G, elevations, drainage, rainfall_mm, model)
    n_flooded = sum(1 for *_, d in G.edges(keys=True, data=True) if d.get("predicted_flooded") == 1)
    n_total = G.number_of_edges()

    risk_level = "Low" if rainfall_mm < 40 else ("Moderate" if rainfall_mm < 90 else "High")
    risk_color = {"Low": "green", "Moderate": "orange", "High": "red"}[risk_level]

    col1, col2, col3 = st.columns(3)
    col1.metric("City-wide risk level", risk_level)
    col2.metric("Roads at flood risk", f"{n_flooded} / {n_total}")
    col3.metric("Expected rain (6h)", f"{rainfall_mm} mm")
    st.markdown(f"**Overall risk indicator:** :{risk_color}[{risk_level}]")

    tab1, tab2 = st.tabs(["\U0001F5FA\uFE0F City Risk Map", "\U0001F9ED Route Planner"])

    with tab1:
        m = draw_network_map(G, center=rn.RAIPUR_CENTER, rainfall_mm=rainfall_mm,
                              user_latlon=(user_lat, user_lon) if user_lat else None)
        st_folium(m, width=None, height=520, key="risk_map")

        colL, colR = st.columns(2)
        with colL:
            st.markdown("### \U0001F30A Nearby rivers & water bodies")
            for r in RIVERS:
                st.write(f"**{r['name']}** -- {r['note']}")
            st.markdown("### \U0001F3E5 Safe shelters")
            for _, s in shelters_df.iterrows():
                st.write(f"**{s['name']}** -- {s['address']} | {s['phone']}")
        with colR:
            st.markdown("### \u26A0\uFE0F Historically flood-prone areas")
            for area in HISTORICAL_FLOOD_AREAS:
                st.write(f"- {area}")
            if user_lat and user_lon:
                nearby_flooded = []
                for u, v, k, d in G.edges(keys=True, data=True):
                    if d.get("predicted_flooded") == 1:
                        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
                        mid_lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
                        dist = rn.haversine_km(user_lat, user_lon, mid_lat, mid_lon)
                        if dist < 2.0:
                            nearby_flooded.append(dist)
                st.markdown("### \U0001F4E2 Alerts near you")
                if nearby_flooded:
                    st.warning(f"{len(nearby_flooded)} road segment(s) predicted flooded within 2km of your location.")
                else:
                    st.success("No high-risk roads predicted within 2km of your current location.")

    with tab2:
        st.markdown("Enter a starting point and destination (any address, landmark, or area name in Raipur).")
        c1, c2 = st.columns(2)
        start_query = c1.text_input("Start", value="Gudhiyari")
        end_query = c2.text_input("Destination", value="AIIMS Raipur")

        if st.button("Find routes", type="primary"):
            with st.spinner("Looking up locations and calculating routes..."):
                start_lat, start_lon = rn.geocode_place(start_query)
                end_lat, end_lon = rn.geocode_place(end_query)

            if not start_lat or not end_lat:
                st.error("Couldn't find one or both locations. Try a more specific place name.")
                st.session_state.pop("route_options", None)
            else:
                import osmnx as ox
                start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
                end_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)
                options = get_route_options(G, start_node, end_node)
                if not options:
                    st.error("No route currently avoids flooding between these two points -- "
                              "or one of them is completely cut off.")
                    st.session_state.pop("route_options", None)
                else:
                    st.session_state["route_options"] = options
                    st.session_state["route_start"] = (start_lat, start_lon)
                    st.session_state["route_end"] = (end_lat, end_lon)
                    st.session_state["tracking"] = False

        options = st.session_state.get("route_options")
        if options:
            st.markdown("#### Choose a route")
            labels = [f"{o['label']} -- {o['distance_km']:.1f} km, avg risk {o['avg_risk']*100:.0f}%"
                      for o in options]
            chosen_idx = st.radio("Route options", range(len(options)), format_func=lambda i: labels[i],
                                   label_visibility="collapsed")
            chosen = options[chosen_idx]
            other_routes = [o["latlon"] for i, o in enumerate(options) if i != chosen_idx]

            start_latlon = st.session_state["route_start"]
            end_latlon = st.session_state["route_end"]

            trackcol1, trackcol2 = st.columns([1, 3])
            tracking = trackcol1.toggle("\U0001F4CD Start live tracking", value=st.session_state.get("tracking", False))
            st.session_state["tracking"] = tracking

            live_user_latlon = None
            if tracking:
                try:
                    from streamlit_autorefresh import st_autorefresh
                    st_autorefresh(interval=6000, key="track_refresh")
                except Exception:
                    pass
                fresh_loc = get_geolocation()
                if fresh_loc and isinstance(fresh_loc, dict) and "coords" in fresh_loc:
                    live_user_latlon = (fresh_loc["coords"]["latitude"], fresh_loc["coords"]["longitude"])
                    remaining_km = rn.haversine_km(live_user_latlon[0], live_user_latlon[1],
                                                     end_latlon[0], end_latlon[1])
                    trackcol2.metric("Distance remaining to destination", f"{remaining_km:.2f} km (straight-line)")
                    st.caption(
                        "\u2139\uFE0F Live tracking here re-checks your real GPS position every ~6 seconds and "
                        "shows straight-line distance remaining -- it doesn't snap to the road or give turn-by-turn "
                        "voice directions like Google Maps, but your position and distance genuinely update as you move."
                    )
                else:
                    trackcol2.info("Waiting for a live GPS fix...")

            st.success(f"\u2705 {chosen['label']} -- approx. {chosen['distance_km']:.1f} km")
            m2 = draw_network_map(G, center=rn.RAIPUR_CENTER, route_latlon=chosen["latlon"],
                                   alt_routes_latlon=other_routes, rainfall_mm=rainfall_mm,
                                   start_latlon=start_latlon, end_latlon=end_latlon,
                                   user_latlon=live_user_latlon)
            st_folium(m2, width=None, height=520, key="route_map")

st.divider()
st.caption(
    "FloodShield prototype -- a learning project. Combines real OpenStreetMap road data, "
    "real elevation data, live weather forecasts, and real reported flood-prone areas with a "
    "machine-learning risk model trained on simulated historical storms. This is a demonstration, "
    "not an official warning system -- always follow guidance from local authorities in a real emergency."
)
