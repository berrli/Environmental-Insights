#!/usr/bin/env python3
"""
build_station_dashboard.py

Creates TWO Kepler.gl HTML dashboards with station locations:

1) UK (ML-HAPPE) stations: one point layer per pollutant
   → book/_static/stations_uk_combined.html

2) Global (ML-HAPPG) stations: one point layer per pollutant
   → book/_static/stations_global_combined.html
"""

import os
import time
import pandas as pd
import geopandas as gpd
from environmental_insights import data as ei_data
import environmental_insights.download as ei_download  # for fallback station-name listing
from keplergl import KeplerGl
from copy import deepcopy

# === USER CONFIGURATION ===
UK_POLLUTANTS      = ["no2", "nox", "no", "o3", "pm10", "pm2p5", "so2"]
GLOBAL_POLLUTANTS  = ["no2", "o3", "pm10", "pm2p5", "so2"]  # NOTE: no 'nox' or 'no' in ML-HAPPG
TARGET_CRS = "EPSG:4326"
MAX_RETRIES = 4
BASE_BACKOFF = 1.5  # seconds

# Fill colors per pollutant (RGB)
COLOR_PALETTE = {
    "no2":   [228,  26,  28],
    "nox":   [ 55, 126, 184],
    "no":    [ 77, 175,  74],
    "o3":    [152,  78, 163],
    "pm10":  [255, 127,   0],
    "pm2p5": [255, 255,  51],
    "so2":   [166,  86,  40]
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "book", "_static"))
os.makedirs(STATIC_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Kepler "point" layer template
# -----------------------------------------------------------------------------
BASE_POINT_LAYER = {
    "id": "",
    "type": "point",
    "config": {
        "dataId": "",
        "label": "",
        "color": [18, 147, 154],
        "columns": {"lat": "latitude", "lng": "longitude", "altitude": None},
        "isVisible": True,
        "visConfig": {
            "radius": 10,
            "fixedRadius": False,
            "opacity": 0.5,
            "outline": False,
            "thickness": 0.5,
            "strokeColor": [0, 0, 0],
            "colorRange": {
                "name": "Global Warming",
                "type": "diverging",
                "category": "Uber",
                "colors": [
                    "#5A1846", "#900C3F", "#C70039",
                    "#E3611C", "#F1920E", "#FFC300"
                ]
            },
            "strokeColorRange": {
                "name": "ColorBrewer Accent",
                "type": "qualitative",
                "category": "ColorBrewer",
                "colors": [
                    "#8DD3C7", "#FFFFB3", "#BEBADA",
                    "#FB8072", "#80B1D3", "#FDB462",
                    "#B3DE69", "#FCCDE5", "#D9D9D9",
                    "#BC80BD", "#CCEBC5", "#FFED6F"
                ]
            },
            "radiusRange": [1, 10],
            "filled": True,
            "textLabel": [{
                "field": {"name": "station", "type": "string"},
                "size": 12,
                "color": [255, 255, 255],
                "anchor": "start",
                "alignment": "center"
            }]
        },
        "hidden": False,
        "textLabel": []
    },
    "visualChannels": {
        "colorField": None,
        "colorScale": "linear",
        "strokeColorField": None,
        "strokeColorScale": "linear",
        "sizeField": None,
        "sizeScale": "linear",
        "heightField": None,
        "heightScale": "linear",
        "radiusField": None,
        "radiusScale": "linear"
    }
}

def build_layers(pollutants):
    layers = []
    for pol in pollutants:
        pol_upper = pol.upper()
        layer_id = f"stations_{pol_upper}_pt"
        layer = deepcopy(BASE_POINT_LAYER)
        layer["id"] = layer_id
        layer["config"]["dataId"] = layer_id
        layer["config"]["label"] = f"{pol_upper} Stations"
        layer["config"]["color"] = COLOR_PALETTE.get(pol, [18, 147, 154])
        layers.append(layer)
    return layers

def build_tooltip_config(pollutants):
    fields = {}
    for pol in pollutants:
        layer_id = f"stations_{pol.upper()}_pt"
        fields[layer_id] = [
            {"name": "station",   "format": None},
            {"name": "pollutant", "format": None}
        ]
    return {"fieldsToShow": fields, "compareMode": False, "enabled": True}

def make_map_config(center_lat, center_lon, zoom, pollutants):
    return {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": build_layers(pollutants),
                "interactionConfig": {
                    "tooltip": build_tooltip_config(pollutants),
                    "brush": {"size": 0.5, "enabled": False},
                    "geocoder": {"enabled": False},
                    "coordinate": {"enabled": False}
                },
                "layerBlending": "normal",
                "splitMaps": [],
                "animationConfig": {"currentTime": None, "speed": 1}
            },
            "mapState": {
                "bearing": 0,
                "dragRotate": False,
                "latitude": center_lat,
                "longitude": center_lon,
                "pitch": 0,
                "zoom": zoom,
                "isSplit": False
            }
        }
    }

# -----------------------------------------------------------------------------
# Retry helpers
# -----------------------------------------------------------------------------
def _retry_call(fn, *args, what: str, dataset: str, **kwargs):
    """Generic retry wrapper with backoff. Returns fn(...) or None after MAX_RETRIES failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = BASE_BACKOFF * (attempt ** 2)
            print(f"[{dataset}] {what}: attempt {attempt}/{MAX_RETRIES} failed with {type(e).__name__}: {e}. "
                  f"Retrying in {wait:.1f}s…")
            time.sleep(wait)
    print(f"[{dataset}] Skipping {what} after {MAX_RETRIES} failed attempts.")
    return None

# -----------------------------------------------------------------------------
# Station collectors (station, pollutant, latitude, longitude)
# -----------------------------------------------------------------------------
def _get_uk_station_names(pol: str) -> list[str]:
    """Use data.get_uk_monitoring_stations; if older signature raises TypeError, call downloader directly."""
    try:
        return ei_data.get_uk_monitoring_stations(pol)
    except TypeError:
        return ei_download.get_training_station_names("ML-HAPPE", pol)

def collect_uk_stations(pollutants):
    records = []
    names = {}
    for pol in pollutants:
        # Fetch the list with retry (network hiccups possible)
        station_names = _retry_call(_get_uk_station_names, pol, what=f"UK station list for {pol}", dataset="ML-HAPPE")
        if not station_names:
            continue
        names[pol] = station_names

    for pol, station_names in names.items():
        for station in station_names:
            gdf = _retry_call(
                ei_data.get_uk_monitoring_station,
                pol, station,
                what=f"UK station {station} ({pol})",
                dataset="ML-HAPPE"
            )
            if gdf is None or gdf.empty:
                continue
            if gdf.crs is None:
                # Helper typically sets a projected CRS; fall back conservatively, then reproject.
                gdf = gdf.set_crs(epsg=3395)
            gdf = gdf.to_crs(TARGET_CRS)
            pt = gdf.geometry.iloc[0]
            records.append({
                "station":   station.replace("_", " ").title(),
                "pollutant": pol.upper(),
                "latitude":  float(pt.y),
                "longitude": float(pt.x),
            })
    return pd.DataFrame(records)

def collect_global_stations(pollutants):
    records = []
    names = {}
    for pol in pollutants:
        station_names = _retry_call(
            ei_data.get_global_monitoring_stations,
            pol,
            what=f"Global station list for {pol}",
            dataset="ML-HAPPG"
        )
        if not station_names:
            continue
        names[pol] = station_names

    for pol, station_names in names.items():
        for station in station_names:
            gdf = _retry_call(
                ei_data.get_global_monitoring_station,
                pol, station,
                what=f"Global station {station} ({pol})",
                dataset="ML-HAPPG"
            )
            if gdf is None or gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=3395)
            gdf = gdf.to_crs(TARGET_CRS)
            pt = gdf.geometry.iloc[0]
            records.append({
                "station":   station.replace("_", " ").title(),
                "pollutant": pol.upper(),
                "latitude":  float(pt.y),
                "longitude": float(pt.x),
            })
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Save map utilities
# -----------------------------------------------------------------------------
def save_stations_map(stations_df: pd.DataFrame, pollutants, *, center_lat, center_lon, zoom, filename):
    out_path = os.path.join(STATIC_DIR, filename)
    kmap = KeplerGl(config=make_map_config(center_lat, center_lon, zoom, pollutants), height=600)

    for pol in pollutants:
        pol_upper = pol.upper()
        subset = stations_df[stations_df["pollutant"] == pol_upper].copy()
        if subset.empty:
            print(f"[warn] No stations found for layer {pol_upper}; layer will be empty.")
        layer_id = f"stations_{pol_upper}_pt"
        kmap.add_data(data=subset, name=layer_id)

    kmap.save_to_html(file_name=out_path)
    print(f"Map saved to: {out_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # --- UK (ML-HAPPE) ---
    uk_df = collect_uk_stations(UK_POLLUTANTS)
    save_stations_map(
        uk_df, UK_POLLUTANTS,
        center_lat=54.0, center_lon=-2.0, zoom=5,
        filename="stations_uk_combined.html"
    )

    # --- Global (ML-HAPPG) ---
    global_df = collect_global_stations(GLOBAL_POLLUTANTS)
    save_stations_map(
        global_df, GLOBAL_POLLUTANTS,
        center_lat=0.0, center_lon=0.0, zoom=1.8,
        filename="stations_global_combined.html"
    )

if __name__ == "__main__":
    main()
