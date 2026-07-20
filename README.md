# 🌊 FloodShield — Raipur Prototype

An ML-powered flood risk prediction and route-planning app, prototyped for
**Raipur, Chhattisgarh**, built with expansion to other Indian cities in mind.

## What it does

1. **Detects your real location** via the browser's geolocation permission,
   and confirms whether FloodShield has data for your city (currently:
   Raipur only)
2. **Pulls Raipur's real road network** live from OpenStreetMap (~4.5km
   radius, core urban area)
3. **Gets real elevation** for every intersection via the free Open-Elevation
   API
4. **Fetches a live rainfall forecast** for the next 6 hours via the
   OpenWeatherMap API (you supply your own free API key)
5. **Estimates drainage quality** per road based on real, documented
   flood-prone neighbourhoods (Gudhiyari, Daganiya, Tatibandh — from local
   news reports)
6. **Predicts flood risk per road** with a Random Forest classifier trained
   on simulated historical storms (**~91% accuracy** on held-out test data)
7. **Free-form route search** — type any two place names in Raipur; the app
   geocodes them, snaps to the real road network, and finds the safest route
   (Dijkstra's algorithm via NetworkX), avoiding high-risk roads
8. **Nearby alerts** — flags flooded road segments within 2km of your
   detected location
9. **Interactive map** — a color-coded flood-risk overlay on the real street
   network

## ⚠️ Why this doesn't use live satellite flood detection

Flood-mapping satellites (e.g. Sentinel-1 radar) revisit any given city only
once every few days — there is no free, real-time "is this exact road
flooded right now" satellite feed, and turning radar imagery into a flood
map is itself a specialised research task. Instead, FloodShield combines
**real live weather forecast data** with **real elevation** and **real,
documented flood-prone areas** — the same core approach real
flood-nowcasting systems use, since continuous live satellite imagery isn't
actually available either.

## Live demo

[Add your Streamlit Community Cloud link here after deploying — see
"Deploying" below]

## How it works

- `raipur_data.py` — real Raipur shelters, rivers, emergency contacts, and
  documented flood-prone area descriptions
- `raipur_network.py` — pulls the real OpenStreetMap road network, gets real
  elevation per intersection, estimates drainage quality from real
  flood-prone hotspots, geocodes place-name searches, and fetches the live
  weather forecast
- `train_model.py` — simulates historical storms using realistic
  elevation/drainage/rainfall relationships, and trains a Random Forest
  classifier to predict flood risk per road
- `app.py` — the Streamlit app: real location detection, live weather, a
  city-wide risk map, and a free-form route planner

## ⚠️ Honesty note on the data

The place names, the Kharun River, and the flood-prone neighbourhoods
(Gudhiyari, Gudhiyari Underbridge, Daganiya, Tatibandh) are **real**, based
on 2026 local news reports of monsoon waterlogging in Raipur and government
sources. The real road network and elevation come from live public APIs
(OpenStreetMap, Open-Elevation). However:

- Drainage-quality values are a **distance-based estimate** from known
  flood-prone hotspots, not a measured engineering survey
- Historical storm training data is **simulated**, not real sensor records
- Rainfall is a **6-hour forecast total**, used as a proxy for storm
  severity — not a calibrated hydrological model

A fully production-grade version would replace the simulated training data
with real historical flood records, and the drainage estimate with an
actual municipal drainage-infrastructure survey.

Sources used for the real-world facts in this prototype:
- ANI / NewKerala news reports on July 2026 Raipur waterlogging
- Raipur district government website (raipur.gov.in)
- Wikipedia: Kharun River
- Public hospital directories (AIIMS Raipur, DKS Hospital)
- OpenStreetMap (road network), Open-Elevation (elevation),
  OpenWeatherMap (live rainfall forecast)

## Running locally

```bash
pip install -r requirements.txt
python train_model.py   # trains and saves flood_model.joblib (only needed once)
streamlit run app.py
```

You'll need a free OpenWeatherMap API key (openweathermap.org/api_keys) for
live rainfall data — without one, the app falls back to a manual rainfall
slider.

## Deploying for free (no local install needed)

1. Push this whole folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub
3. Click "New app", pick this repo, and set the main file to `app.py`
4. Deploy — you'll get a public URL like
   `https://floodshield-raipur.streamlit.app`
5. On first load, allow the location permission prompt, and paste in your
   OpenWeatherMap key in the sidebar for live rain data

## Roadmap / how this scales to other cities

- Add a `<city>_network.py` (road network centerpoint + flood-prone
  hotspots) and `<city>_data.py` (shelters, rivers, emergency contacts)
  following the same structure as the Raipur versions
- Extend `SUPPORTED_CITIES` in `app.py` and route the detected city to the
  right data module
- Replace the simulated storm-training data with real historical flood
  records where available, city by city

## Tech stack

Python, Streamlit, scikit-learn (Random Forest), NetworkX (graph routing),
OSMnx (real road network), Folium (interactive maps), OpenWeatherMap API
(live weather), Open-Elevation API, Nominatim (geocoding), pandas/numpy.
