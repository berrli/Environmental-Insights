#!/usr/bin/env python3
"""
Builds three KeplerGl maps:
  • UK SynthHAPPE (typical-day) — NO₂
  • UK ML-HAPPE (real-time 2018) — NO₂
  • Global ML-HAPPG (real-time 2022) — O₃

Saved to ../book/_static/.
"""

import os
import copy
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import transform
from keplergl import KeplerGl

# EI helpers
from environmental_insights import data as ei_data

# === CONFIGURE HERE ===
MONTH               = 1
DAY_OF_WEEK         = "Friday"
HOUR                = 9

# Separate real-time targets:
UK_REALTIME_TIME      = "2018-01-01_090000"  # ML-HAPPE (UK) -> underscore
GLOBAL_REALTIME_TIME  = "2022-07-02_080000"  # ML-HAPPG (Global) -> will normalize to space on call

# UK variables (NO2)
VAR_NAME_UK       = "no2_Prediction_Mean"
VAR_NEAT_UK       = "Nitrogen Dioxide Prediction Mean"

# Global variables (O3)
VAR_NAME_GLOBAL   = "o3_Prediction_Mean"
VAR_NEAT_GLOBAL   = "Ozone Prediction Mean"
# ======================

# Pretty console output
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', None)

# KeplerGl base config (we'll tweak per-map in save_map)
BASE_CONFIG = {
    'version': 'v1',
    'config': {
        'visState': {
            'filters': [],
            'layers': [{
                'id': 'gyunm4n',
                'type': 'geojson',
                'config': {
                    'dataId': None,
                    'label': None,
                    'color': [77, 193, 156],
                    'highlightColor': [252, 242, 26, 255],
                    'columns': {'geojson': 'geometry'},
                    'isVisible': True,
                    'visConfig': {
                        'opacity': 0.2,
                        'strokeOpacity': 0.8,
                        'thickness': 0.5,
                        'strokeColor': [119, 110, 87],
                        'colorRange': {
                            'name': 'Global Warming',
                            'type': 'sequential',
                            'category': 'Uber',
                            'colors': [
                                '#5A1846', '#900C3F', '#C70039',
                                '#E3611C', '#F1920E', '#FFC300'
                            ]
                        },
                        'strokeColorRange': {
                            'name': 'Global Warming',
                            'type': 'sequential',
                            'category': 'Uber',
                            'colors': [
                                '#5A1846', '#900C3F', '#C70039',
                                '#E3611C', '#F1920E', '#FFC300'
                            ]
                        },
                        'radius': 10,
                        'sizeRange': [0, 10],
                        'radiusRange': [0, 50],
                        'heightRange': [0, 500],
                        'elevationScale': 5,
                        'enableElevationZoomFactor': True,
                        'stroked': False,
                        'filled': True,
                        'enable3d': False,
                        'wireframe': False
                    },
                    'hidden': False,
                    'textLabel': [{
                        'field': None,
                        'color': [255, 255, 255],
                        'size': 18,
                        'offset': [0, 0],
                        'anchor': 'start',
                        'alignment': 'center'
                    }]
                },
                'visualChannels': {
                    # We'll inject the correct column name per map
                    'colorField': {'name': 'VALUE_LABEL_PLACEHOLDER', 'type': 'real'},
                    'colorScale': 'quantile',
                    'strokeColorField': None,
                    'strokeColorScale': 'quantile',
                    'sizeField': None,
                    'sizeScale': 'linear',
                    'heightField': None,
                    'heightScale': 'linear',
                    'radiusField': None,
                    'radiusScale': 'linear'
                }
            }],
            'interactionConfig': {
                'tooltip': {'fieldsToShow': {}, 'compareMode': False, 'compareType': 'absolute', 'enabled': True},
                'brush': {'size': 0.5, 'enabled': False},
                'geocoder': {'enabled': False},
                'coordinate': {'enabled': False}
            },
            'layerBlending': 'normal',
            'splitMaps': [],
            'animationConfig': {'currentTime': None, 'speed': 1}
        },
        # UK-centric default view; global view can override per map
        'mapState': {
            'bearing': 0,
            'dragRotate': False,
            'latitude': 52.69738599316781,
            'longitude': -0.29600986227730636,
            'pitch': 0,
            'zoom': 5,
            'isSplit': False
        }
    }
}

def load_uk_grid():
    gdf = ei_data.get_uk_grids()
    return gdf[["UK_Model_Grid_ID", "geometry"]]

def load_global_grid():
    gdf = ei_data.get_global_grids()
    return gdf[["Global_Model_Grid_ID", "geometry"]]

def prepare_and_merge(
    df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    id_col: str,
    value_col: str,
    value_label: str,
    simplify_tol: float = 0.0005,
    quantize_digits: int = 4
) -> gpd.GeoDataFrame:
    """Generic polygon merge/clean for UK or Global."""
    if id_col not in df.columns:
        raise KeyError(f"Expected ID column '{id_col}' not found in dataframe: {df.columns.tolist()}")
    if value_col not in df.columns:
        raise KeyError(f"Expected value column '{value_col}' not found in dataframe: {df.columns.tolist()}")

    df2 = df[[id_col, value_col]].dropna(subset=[value_col]).copy()
    df2[value_col] = df2[value_col].astype(np.float16)

    merged = grid_gdf.merge(df2, on=id_col, how="inner")
    if merged.crs != "EPSG:4326":
        merged = merged.to_crs(epsg=4326)

    merged["geometry"] = merged.geometry.simplify(tolerance=simplify_tol, preserve_topology=True)

    def _quantize(geom):
        return transform(lambda x, y: (round(x, quantize_digits), round(y, quantize_digits)), geom)

    merged["geometry"] = merged.geometry.apply(_quantize)
    merged = merged.rename(columns={value_col: value_label})
    return merged[["geometry", value_label]]

def save_map(
    gdf: gpd.GeoDataFrame,
    title: str,
    filename: str,
    *,
    value_label: str,
    map_center=None,
    map_zoom=None
):
    """
    Create a KeplerGl map for `gdf`, save to `filename`, and print confirmation.
    Allows overriding map center/zoom.
    """
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../book/_static'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    conf = copy.deepcopy(BASE_CONFIG)
    conf['config']['visState']['layers'][0]['config']['dataId'] = title
    conf['config']['visState']['layers'][0]['config']['label'] = title
    # Inject correct colorField & tooltip for this dataset
    conf['config']['visState']['layers'][0]['visualChannels']['colorField'] = {
        'name': value_label, 'type': 'real'
    }
    conf['config']['visState']['interactionConfig']['tooltip']['fieldsToShow'] = {
        title: [{'name': value_label, 'format': None}]
    }

    # Optional map view override
    if map_center is not None:
        conf['config']['mapState']['latitude'] = map_center[0]
        conf['config']['mapState']['longitude'] = map_center[1]
    if map_zoom is not None:
        conf['config']['mapState']['zoom'] = map_zoom

    kmap = KeplerGl(config=conf)
    kmap.add_data(data=gdf, name=title)
    kmap.save_to_html(file_name=out_path)
    print(f"{title} map saved to {out_path}")

def main():
    # ---------- UK (SynthHAPPE Output) ----------
    grid_uk = load_uk_grid()
    typ_df = ei_data.air_pollution_concentration_typical_day_real_time_united_kingdom(
        MONTH, DAY_OF_WEEK, HOUR, data_type="Output"
    )
    typ_prep = prepare_and_merge(
        typ_df, grid_uk,
        id_col="UK_Model_Grid_ID",
        value_col=VAR_NAME_UK,
        value_label=VAR_NEAT_UK
    )
    save_map(
        typ_prep,
        title="Typical Day NO₂ Prediction",
        filename="synthhappe_no2.html",
        value_label=VAR_NEAT_UK
    )

    # ---------- UK (ML-HAPPE Output, 2018 underscore) ----------
    rt_df_uk = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom(
        UK_REALTIME_TIME,  # underscore stays for UK
        data_type="Output"
    )
    rt_prep_uk = prepare_and_merge(
        rt_df_uk, grid_uk,
        id_col="UK_Model_Grid_ID",
        value_col=VAR_NAME_UK,
        value_label=VAR_NEAT_UK
    )
    save_map(
        rt_prep_uk,
        title="Real-Time NO₂ Prediction (UK)",
        filename="mlhappe_no2.html",
        value_label=VAR_NEAT_UK
    )

    # ---------- GLOBAL (ML-HAPPG Output, 2022; using O3) ----------
    grid_glob = load_global_grid()
    # Normalize underscore->space for ML-HAPPG timestamp
    rt_df_glob = ei_data.air_pollution_concentration_complete_set_real_time_global(
        GLOBAL_REALTIME_TIME,
        data_type="Output"
    )
    rt_prep_glb = prepare_and_merge(
        rt_df_glob, grid_glob,
        id_col="Global_Model_Grid_ID",
        value_col=VAR_NAME_GLOBAL,
        value_label=VAR_NEAT_GLOBAL
    )

    # Global view
    save_map(
        rt_prep_glb,
        title="Real-Time O₃ Prediction (Global)",
        filename="mlhappg_o3.html",
        value_label=VAR_NEAT_GLOBAL,
        map_center=(0.0, 0.0),
        map_zoom=1.8
    )

if __name__ == "__main__":
    main()
