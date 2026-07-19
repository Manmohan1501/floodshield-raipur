"""
FloodShield -- Raipur Prototype
--------------------------------
An ML-powered flood risk & evacuation app, prototyped for Raipur,
Chhattisgarh. Built to be extended city-by-city later (see README).
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium
import joblib
import os

from raipur_data import (
    city_nodes, roads_df, shelters_df, HISTORICAL_FLOOD_AREAS,
    EMERGENCY_CONTACTS, RIVERS, elev_lookup,
)

st.set_page_config(page_title="FloodShield -- Raipur", page_icon="\U0001F30A", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_model.joblib")

# ---------------------------------------------------------------------
# Load (or lazily train) the model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Fallback: train on the fly if the pickle wasn't shipped
    import train_model  # noqa: F401  (running this trains + saves the model)
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------------------------------------------------------------
# Sidebar -- simulated "location detection"
# ---------------------------------------------------------------------
st.sidebar.title("\U0001F30A FloodShield")
st.sidebar.caption("Prototype build -- currently live for **Raipur, Chhattisgarh** only.")
st.sidebar.success("\U0001F4CD Detected city: **Raipur, Chhattisgarh**")
st.sidebar.caption(
    "In a real deployed app, this would come from the phone's location "
    "permission. In this prototype, Raipur is preselected since it's the "
    "only city with data loaded so far. The app is designed so more "
    "cities can be added later -- see README."
)

mode = st.sidebar.radio(
    "Mode",
    ["Normal Conditions", "Heavy Rainfall / Flood Alert"],
    help="Switch to see how the app changes during an active flood event.",
)

st.sidebar.divider()
st.sidebar.subheader("\U0001F6A8 Emergency Contacts")
for c in EMERGENCY_CONTACTS:
    st.sidebar.write(f"**{c['service']}**: {c['number']}")

st.sidebar.divider()
st.sidebar.caption(
    "\u26A0\uFE0F Coordinates, elevations, and drainage scores in this "
    "prototype are approximate estimates for demonstration -- not "
    "official survey data. See README for data sources and limitations."
)

# ---------------------------------------------------------------------
# Helper: build the road graph given a rainfall amount
# ---------------------------------------------------------------------
def predict_flood_risk(rainfall_mm: float) -> pd.DataFrame:
    roads = roads_df.copy()
    roads["rainfall_mm"] = rainfall_mm
    X = roads[["rainfall_mm", "avg_elevation_m", "drainage_quality"]]
    roads["flood_risk_prob"] = model.predict_proba(X)[:, 1]
    roads["predicted_flooded"] = model.predict(X)
    return roads


def build_graph(roads: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, road in roads.iterrows():
        if road.predicted_flooded == 1:
            if road.flood_risk_prob > 0.85:
                continue  # too dangerous, remove entirely
            cost = 1 + road.flood_risk_prob * 20
        else:
            cost = 1
        G.add_edge(int(road.node_a), int(road.node_b), weight=cost, risk=road.flood_risk_prob,
                   flooded=int(road.predicted_flooded))
    return G


def node_latlon(node_id):
    row = city_nodes[city_nodes.node_id == node_id].iloc[0]
    return row.lat, row.lon


def draw_map(roads: pd.DataFrame, route=None, start=None, end=None):
    center = [city_nodes.lat.mean(), city_nodes.lon.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # Roads
    for _, road in roads.iterrows():
        a_lat, a_lon = node_latlon(road.node_a)
        b_lat, b_lon = node_latlon(road.node_b)
        if road.predicted_flooded == 1:
            color = "red" if road.flood_risk_prob > 0.85 else "orange"
            weight = 5
        else:
            color = "#3b82f6"
            weight = 3
        folium.PolyLine([[a_lat, a_lon], [b_lat, b_lon]], color=color, weight=weight,
                         opacity=0.8).add_to(m)

    # Route highlight
    if route:
        pts = [node_latlon(n) for n in route]
        folium.PolyLine(pts, color="limegreen", weight=7, opacity=0.95,
                         tooltip="Recommended evacuation route").add_to(m)

    # Nodes
    for _, node in city_nodes.iterrows():
        popup = f"{node['name']} (elev ~{node.elevation_m}m)"
        color = "gray"
        icon = "info-sign"
        if node.type == "shelter":
            color, icon = "green", "plus-sign"
        elif node.type == "flood_prone":
            color, icon = "red", "warning-sign"
        elif node.node_id == start:
            color, icon = "blue", "user"
        elif node.node_id == end:
            color, icon = "black", "star"
        folium.Marker([node.lat, node.lon], popup=popup, tooltip=node['name'],
                       icon=folium.Icon(color=color, icon=icon)).add_to(m)
    return m


st.title("\U0001F30A FloodShield -- Raipur")

# =======================================================================
# NORMAL CONDITIONS MODE
# =======================================================================
if mode == "Normal Conditions":
    st.subheader("Current status: Normal conditions")

    baseline_roads = predict_flood_risk(rainfall_mm=20)  # light rainfall baseline
    overall_risk = baseline_roads.flood_risk_prob.mean()
    risk_label = "Low" if overall_risk < 0.35 else ("Moderate" if overall_risk < 0.6 else "High")
    risk_color = {"Low": "green", "Moderate": "orange", "High": "red"}[risk_label]

    col1, col2, col3 = st.columns(3)
    col1.metric("City-wide flood risk level", risk_label)
    col2.metric("Rivers nearby", len(RIVERS))
    col3.metric("Registered shelters", len(shelters_df))

    st.markdown(f"**Overall risk indicator:** :{risk_color}[{risk_label}] (baseline light-rain conditions)")

    map_obj = draw_map(baseline_roads)
    st_folium(map_obj, width=None, height=520)

    left, right = st.columns(2)
    with left:
        st.markdown("### \U0001F30A Nearby rivers & water bodies")
        for r in RIVERS:
            st.write(f"**{r['name']}** -- {r['note']}")

        st.markdown("### \U0001F3E5 Safe shelters")
        for _, s in shelters_df.iterrows():
            st.write(f"**{s['name']}**")
            st.caption(f"{s['address']} | {s['phone']} | Capacity: {s['capacity_estimate']}")

    with right:
        st.markdown("### \u26A0\uFE0F Historically flood-prone areas")
        for area in HISTORICAL_FLOOD_AREAS:
            st.write(f"- {area}")

        st.markdown("### \U0001F4DE Emergency contacts")
        for c in EMERGENCY_CONTACTS:
            st.write(f"**{c['service']}**: {c['number']}")

# =======================================================================
# FLOOD ALERT MODE
# =======================================================================
else:
    st.subheader("\U0001F6A8 Heavy Rainfall / Flood Alert Mode")

    rainfall_mm = st.slider(
        "Simulated incoming storm rainfall (mm)", min_value=10, max_value=200, value=110, step=5,
        help="Drag to simulate different storm intensities and watch the risk map and route update.",
    )

    roads = predict_flood_risk(rainfall_mm)
    G = build_graph(roads)

    n_flooded = int(roads.predicted_flooded.sum())
    n_total = len(roads)
    st.error(f"\U0001F6A8 Live prediction: {n_flooded} of {n_total} monitored roads are at high risk of flooding "
             f"at {rainfall_mm}mm rainfall.")

    start_options = {row['name']: row.node_id for _, row in city_nodes.iterrows()}
    default_start = "Gudhiyari" if "Gudhiyari" in start_options else list(start_options.keys())[0]
    start_name = st.selectbox("Your current location", list(start_options.keys()),
                               index=list(start_options.keys()).index(default_start))
    start_node = start_options[start_name]

    shelter_options = {row['name']: row.node_id for _, row in shelters_df.iterrows()}
    end_name = st.selectbox("Preferred shelter / safe zone", list(shelter_options.keys()))
    end_node = shelter_options[end_name]

    route = None
    if start_node not in G:
        st.error(f"\u274C {start_name} is completely cut off -- no surviving safe roads from this location.")
    elif end_node not in G:
        st.warning(f"\u26A0\uFE0F {end_name} is unreachable -- every route there is predicted to flood. "
                   f"Showing nearest reachable alternatives instead.")
        reachable = nx.single_source_dijkstra_path_length(G, start_node, weight="weight")
        alt = sorted(reachable.items(), key=lambda x: x[1])[:5]
        st.markdown("**Nearest reachable safe points:**")
        for node_id, cost in alt:
            name = city_nodes[city_nodes.node_id == node_id].iloc[0]['name']
            st.write(f"- {name} (safety cost {cost:.1f})")
    else:
        try:
            route = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
            cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight="weight")
            route_names = [city_nodes[city_nodes.node_id == n].iloc[0]['name'] for n in route]
            st.success("\u2705 Recommended evacuation route: " + " \u2192 ".join(route_names))
            st.caption(f"Route safety cost: {cost:.1f} (lower = safer / faster)")
        except nx.NetworkXNoPath:
            st.error(f"\u274C No path currently exists from {start_name} to {end_name} -- flooding has split the city.")

    map_obj = draw_map(roads, route=route, start=start_node, end=end_node)
    st_folium(map_obj, width=None, height=520)

    st.markdown("### \U0001F6D1 Roads to avoid right now")
    risky = roads[roads.predicted_flooded == 1].sort_values("flood_risk_prob", ascending=False)
    if len(risky) == 0:
        st.write("No roads currently predicted at high flood risk.")
    else:
        for _, r in risky.iterrows():
            a_name = city_nodes[city_nodes.node_id == r.node_a].iloc[0]['name']
            b_name = city_nodes[city_nodes.node_id == r.node_b].iloc[0]['name']
            st.write(f"- **{a_name} \u2194 {b_name}** -- flood risk {r.flood_risk_prob*100:.0f}%")

    st.markdown("### \U0001F4E2 Alerts")
    st.warning(
        f"Heavy rainfall ({rainfall_mm}mm) detected in the Raipur area. Avoid low-lying roads near "
        f"Gudhiyari, the Gudhiyari Underbridge, and Daganiya. Move to higher ground or a listed shelter "
        f"if you are in an affected zone."
    )

st.divider()
st.caption(
    "FloodShield prototype -- built as a learning project. Flood predictions are generated by a "
    "machine-learning model trained on simulated historical storms using real place names and "
    "approximate real geography for Raipur. This is a demonstration, not an official warning system. "
    "For real emergencies, always follow official guidance from local authorities."
)
