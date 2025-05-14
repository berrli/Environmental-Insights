import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import environmental_insights.models as ei_models
import environmental_insights.download as ei_download
import environmental_insights.variables as variables


class TestModelFunctions(unittest.TestCase):

    @patch("environmental_insights.download.download")
    def test_ensure_download_calls_downloader(self, mock_download):
        ei_models.ensure_download("ML-HAPPE", "Input", timestamp="2023-01-01_080000")
        mock_download.assert_called_once_with(
            dataset="ML-HAPPE",
            data_type="Input",
            timestamp="2023-01-01_080000",
            month=None,
            day=None,
            hour=None,
            model_level=None,
            pollutant=None,
            station=None,
            token=None,
            output_dir=None,
        )

    @patch("builtins.open", new_callable=mock_open, read_data='{"boosting_type": "gbdt"}')
    @patch("lightgbm.Booster")
    def test_load_lgbm_model_from_txt_success(self, mock_booster, mock_file):
        mock_booster.return_value.num_feature.return_value = 10

        model = ei_models.load_lgbm_model_from_txt("dummy_booster.txt", "dummy_params.json")

        self.assertIsInstance(model, lgb.LGBMRegressor)
        self.assertEqual(model.n_features_in_, 10)
        mock_booster.assert_called_once_with(model_file="dummy_booster.txt")
        mock_file.assert_called_with("dummy_params.json", "r")

    @patch("environmental_insights.download.download_file")
    @patch("environmental_insights.models.load_lgbm_model_from_txt")
    def test_load_model_united_kingdom(self, mock_load_model, mock_download_file):
        mock_load_model.return_value = lgb.LGBMRegressor()

        model = ei_models.load_model_united_kingdom("mean", "no2", "Forecasting_Models", token="abc123")

        base_url = ei_download.BASE_URLS['ML-HAPPE']
        expected_booster_url = f"{base_url}Models/mean/no2/All_Stations/no2_Forecasting_Models/model_booster.txt"
        expected_params_url = f"{base_url}Models/mean/no2/All_Stations/no2_Forecasting_Models/model_params.json"

        expected_local_dir = Path("environmental_insights/environmental_insights_models/ML-HAPPE/Models/mean/no2/All_Stations/no2_Forecasting_Models")

        mock_download_file.assert_any_call(expected_booster_url, output_dir=expected_local_dir, token="abc123")
        mock_download_file.assert_any_call(expected_params_url, output_dir=expected_local_dir, token="abc123")
        self.assertIsInstance(model, lgb.LGBMRegressor)

    @patch("environmental_insights.models.load_model_united_kingdom")
    def test_load_model_global(self, mock_load_model_uk):
        mock_load_model_uk.return_value = "mock_model"

        result = ei_models.load_model_global("0.5", "o3", "Global_Models")
        self.assertEqual(result, "mock_model")
        mock_load_model_uk.assert_called_once_with("0.5", "o3", "Global_Models", None)

    def test_get_model_feature_vector(self):
        variables.featureVectorSubsets = {"test_model": ["feature1", "feature2"]}
        features = ei_models.get_model_feature_vector("test_model")
        self.assertEqual(features, ["feature1", "feature2"])

    def test_make_concentration_predictions_united_kingdom(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.log(np.array([1.5, 2.5, 3.5]) + 1e-7)

        observation_data = pd.DataFrame({
            "UK_Model_Grid_ID": [101, 102, 103],
            "feature1": [10, 20, 30],
            "feature2": [0.1, 0.2, 0.3],
        })

        feature_names = ["feature1", "feature2"]

        output_df = ei_models.make_concentration_predictions_united_kingdom(mock_model, observation_data, feature_names)

        self.assertIn("UK_Model_Grid_ID", output_df.columns)
        self.assertIn("Model Prediction", output_df.columns)
        self.assertTrue(np.allclose(output_df["Model Prediction"], [1.5, 2.5, 3.5]))
