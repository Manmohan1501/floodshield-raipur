# \U0001F30A FloodShield -- Raipur Prototype

An ML-powered flood risk prediction and evacuation guidance app,
prototyped for **Raipur, Chhattisgarh**, built with expansion to other
Indian cities in mind.

## Concept

FloodShield detects the user's city and shows:

**During normal conditions:**
- City-wide flood risk level
- Nearby rivers and water bodies
- Safe shelters (with address & contact)
- Emergency contacts
- Historically flood-prone areas

**During heavy rainfall / a flood event:**
- Live ML-based flood risk prediction per road
- Recommended evacuation route (with automatic fallback if the chosen
  safe zone becomes unreachable)
- Roads to avoid
- Alerts

This repo implements a **working prototype for one city (Raipur)**. The
code is structured so more cities can be added later by adding a new
`<city>_data.py` file with that city's nodes/roads/shelters -- the model
training and app logic are city-agnostic.

## Live demo

[Add your Streamlit Community Cloud link here after deploying -- see
"Deploying" below]

## How it works

1. `raipur_data.py` -- real Raipur place names (Gudhiyari, Daganiya,
   Tatibandh, Kharun River, AIIMS Raipur, etc.), connected by roads,
   with elevation and drainage-quality estimates.
2. `train_model.py` -- simulates historical storms using those real
   locations' geography, and trains a Random Forest classifier to
   predict flood risk per road (currently **~89% accuracy** on held-out
   test data).
3. `app.py` -- a Streamlit app with two modes (Normal / Flood Alert)
   that uses the trained model's live predictions to:
   - Color-code roads by flood risk on an interactive map
   - Find the safest evacuation route with NetworkX (Dijkstra's
     algorithm), avoiding high-risk roads
   - Fall back to the nearest reachable safe point if the chosen
     shelter becomes unreachable

## \u26A0\uFE0F Honesty note on the data

The place names, the Kharun River, and the flood-prone neighbourhoods
(Gudhiyari, Gudhiyari Underbridge, Daganiya, Tatibandh) are **real**,
based on 2026 local news reports of monsoon waterlogging in Raipur and
government sources. However:

- Exact coordinates are **approximate**, not surveyed GIS data
- Elevation and drainage-quality values are **estimates** for
  demonstration, not measured data
- Historical storm data is **simulated**, not real sensor records

A production version would replace these with real elevation data
(e.g. a government DEM/GIS source), real road network data (e.g.
OpenStreetMap), and real historical flood records.

Sources used for the real-world facts in this prototype:
- ANI / NewKerala news reports on July 2026 Raipur waterlogging
- Raipur district government website (raipur.gov.in)
- Wikipedia: Kharun River
- Public hospital directories (AIIMS Raipur, DKS Hospital)

## Running locally

```bash
pip install -r requirements.txt
python train_model.py   # trains and saves flood_model.joblib (only needed once)
streamlit run app.py
```

## Deploying for free (no local install needed)

1. Push this whole folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub
3. Click "New app", pick this repo, and set the main file to `app.py`
4. Deploy -- you'll get a public URL like
   `https://floodshield-raipur.streamlit.app`

## Roadmap / how this scales to other cities

- Add a `<city>_data.py` file following the same structure as
  `raipur_data.py`
- Add real elevation/road data for that city
- Extend the sidebar's "detected city" logic to switch between city
  data files
- Replace the simulated storm history with real historical/live
  weather API data where available

## Tech stack

Python, Streamlit, scikit-learn (Random Forest), NetworkX (graph
routing), Folium (interactive maps), pandas/numpy.
