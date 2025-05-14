import os
import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import lightgbm as lgb

import environmental_insights.variables as variables
import environmental_insights.download as ei_download  # unified downloader


# Constants for directory structure
MODEL_ROOT = Path("environmental_insights/environmental_insights_models")
DATA_ROOT = Path("environmental_insights/environmental_insights_data")
MODEL_CATEGORIES = [
    "All",
    "Emissions_Models",
    "Forecasting_Models",
    "Forecasting_Transport_and_Emissions_Models",
    "Geographic_Models",
    "Global_Models",
    "Metrological_Models",
    "Remote_Sensing_Models",
    "Temporal_Models",
    "Transport_Infrastructure_Models",
    "Transport_Use_Models",
]


def ensure_download(
    dataset: str,
    data_type: str,
    *,
    timestamp: Optional[str] = None,
    month: Optional[Union[int, str]] = None,
    day: Optional[str] = None,
    hour: Optional[Union[int, str]] = None,
    model_level: Optional[str] = None,
    pollutant: Optional[str] = None,
    station: Optional[str] = None,
    token: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Ensure the requested file(s) exist locally by invoking the unified downloader.
    """
    ei_download.download(
        dataset=dataset,
        data_type=data_type,
        timestamp=timestamp,
        month=month,
        day=day,
        hour=hour,
        model_level=model_level,
        pollutant=pollutant,
        station=station,
        token=token,
        output_dir=output_dir,
    )


def load_lgbm_model_from_txt(booster_filepath, params_filepath):
    try:
        # Load parameters
        with open(params_filepath, "r") as f:
            params = json.load(f)
        
        # Load Booster
        booster = lgb.Booster(model_file=booster_filepath)

        # Reconstruct model
        model = lgb.LGBMRegressor(**params)
        model._Booster = booster
        model.__sklearn_is_fitted__ = lambda: True
        
        # Correctly set internal attributes
        model.n_features_in_ = booster.num_feature()
        model._n_features = booster.num_feature()
        # Do NOT try to set feature_name_, it's read-only

        print(f"Model loaded successfully from {booster_filepath} and {params_filepath}")
        return model

    except Exception as e:
        print(f"Failed to load the model: {e}")
        return None
    
def load_model_united_kingdom(
    model_level: str,
    pollutant: str,
    model_category: str,
    token: Optional[str] = None
) -> lgb.LGBMRegressor:
    """
    Load a pretrained UK air pollution LGBM model for a specific category,
    downloading only the required booster + params.

    Parameters:
    - model_level: one of ['0.05','0.5','0.95','mean']
    - pollutant: one of ['no','no2','nox','o3','pm10','pm2p5','so2']
    - model_category: one of MODEL_CATEGORIES or 'All'
    """
    # Correctly handle the special "All" category
    if model_category == "All":
        category_path = f"Models/{model_level}/{pollutant}/All_Stations/{pollutant}"
    else:
        category_path = f"Models/{model_level}/{pollutant}/All_Stations/{pollutant}_{model_category}"

    base_url = ei_download.BASE_URLS['ML-HAPPE']
    booster_url = f"{base_url}{category_path}/model_booster.txt"
    params_url = f"{base_url}{category_path}/model_params.json"

    local_dir = MODEL_ROOT / "ML-HAPPE" / category_path
    local_dir.mkdir(parents=True, exist_ok=True)

    booster_file = local_dir / "model_booster.txt"
    params_file = local_dir / "model_params.json"

    # Download only missing files
    if not booster_file.is_file():
        ei_download.download_file(str(booster_url), output_dir=local_dir, token=token)
    if not params_file.is_file():
        ei_download.download_file(str(params_url), output_dir=local_dir, token=token)

    return load_lgbm_model_from_txt(booster_file, params_file)









def load_model_global(
    model_level: str,
    pollutant: str,
    model_category: str,
    token: Optional[str] = None
) -> lgb.LGBMRegressor:
    """
    Load a pretrained Global air pollution LGBM model for a specific category,
    downloading only the required booster + params.
    """
    # Uses same ML-HAPPE structure
    return load_model_united_kingdom(model_level, pollutant, model_category, token)


def get_model_feature_vector(model_type: str) -> List[str]:
    return variables.featureVectorSubsets[model_type]

def make_concentration_predictions_united_kingdom(
    estimating_model: lgb.LGBMRegressor,
    observation_data: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Predict concentrations using a trained LGBM model.
    """
    df = observation_data.copy()

    X = df[feature_names].copy()

    # Prediction
    preds = estimating_model.predict(X)

    # Transform back from log space
    preds = np.exp(preds) - 1e-7

    # Build output DataFrame
    out = df[["UK_Model_Grid_ID"]].copy()
    out["Model Prediction"] = preds

    return out
