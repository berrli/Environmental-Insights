import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
_ = np.array([1])  # Access C-API

import environmental_insights.air_pollution_functions as ei_air_pollution_functions
import environmental_insights.data as ei_data


class TestAirPollutionFunctions(unittest.TestCase):

    def setUp(self):
        # Simplified dummy polygons for UK and Global grids
        dummy_geom = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 3

        self.uk_grids = gpd.GeoDataFrame({
            "UK_Model_Grid_ID": [1, 2, 3],
            "geometry": dummy_geom
        }, crs="EPSG:4326")

        self.global_grids = gpd.GeoDataFrame({
            "Global Model Grid ID": [100, 200, 300],
            "geometry": dummy_geom
        }, crs="EPSG:4326")

        # UK dataset
        self.uk_complete_dataset = pd.DataFrame({
            "UK_Model_Grid_ID": [1, 2, 3],
            "no2_Prediction_Mean": [10, 20, 30],
        })

        # Global dataset with AQI columns
        self.global_complete_dataset = pd.DataFrame({
            "Global Model Grid ID": [100, 200, 300],
            "no2": [40, 50, 60],
        })

        # Convert to GeoDataFrame by merging with global_grids
        self.air_pollution_DF_global = self.global_grids.merge(
            self.global_complete_dataset, on="Global Model Grid ID"
        )

        # AQI DataFrames for UK (8am/9am)
        self.air_pollution_DF_8am = self.uk_grids.copy()
        self.air_pollution_DF_8am["no2_Prediction_Mean"] = [10, 20, 30]
        self.air_pollution_DF_8am["no2 AQI"] = [1, 2, 3]

        self.air_pollution_DF_9am = self.uk_grids.copy()
        self.air_pollution_DF_9am["no2_Prediction_Mean"] = [15, 25, 35]
        self.air_pollution_DF_9am["no2 AQI"] = [2, 3, 4]

        self.air_pollutants = ["no2", "o3", "pm10", "pm2p5", "so2"]

    @patch("environmental_insights.data.air_pollution_concentration_complete_set_real_time_global")
    def test_load_global_dataset(self, mock_global_data):
        mock_global_data.return_value = self.global_complete_dataset
        result = ei_data.air_pollution_concentration_complete_set_real_time_global("07-02-2022_080000")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ["Global Model Grid ID", "no2"])

    @patch("environmental_insights.air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index")
    def test_global_air_quality_index_calculation(self, mock_func):
        mock_func.return_value = pd.DataFrame({
            "Global Model Grid ID": [100, 200, 300],
            "no2 AQI": [2, 3, 4],
            "no2 Air Quality Index AQI Band": ["Low", "Moderate", "High"],
        })
        result_df = ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(
            self.global_complete_dataset, "no2", "no2"
        )
        self.assertIn("no2 AQI", result_df.columns)
        self.assertIn("no2 Air Quality Index AQI Band", result_df.columns)

    @patch("environmental_insights.air_pollution_functions.visualise_air_pollution_daily_air_quality_index")
    def test_visualise_air_pollution_daily_air_quality_index_global(self, mock_func):
        ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_index(
            self.air_pollution_DF_global, "no2 AQI", "output_global_index"
        )
        mock_func.assert_called_once()

    @patch("environmental_insights.air_pollution_functions.visualise_air_pollution_daily_air_quality_bands")
    def test_visualise_air_pollution_daily_air_quality_bands_global(self, mock_func):
        ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands(
            self.air_pollution_DF_global, "no2 Air Quality Index AQI Band", "output_global_band"
        )
        mock_func.assert_called_once()

    def test_change_in_concentrations_visulisation_incorrect(self):
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_concentrations_visulisation(
                "invalid", self.air_pollution_DF_9am, "no2_Prediction_Mean", "output"
            )
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_concentrations_visulisation(
                self.air_pollution_DF_8am, "invalid", "no2_Prediction_Mean", "output"
            )
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.change_in_concentrations_visulisation(
                self.air_pollution_DF_8am.drop(columns=["no2_Prediction_Mean"]),
                self.air_pollution_DF_9am, "no2_Prediction_Mean", "output"
            )

    @patch("environmental_insights.air_pollution_functions.change_in_concentrations_visulisation")
    def test_change_in_concentrations_visulisation_correct(self, mock_func):
        ei_air_pollution_functions.change_in_concentrations_visulisation(
            self.air_pollution_DF_8am, self.air_pollution_DF_9am, "no2_Prediction_Mean", "output"
        )
        mock_func.assert_called_once()

    def test_change_in_aqi_visulisation_incorrect(self):
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_aqi_visulisation(
                "invalid", self.air_pollution_DF_9am, "no2 AQI", "output"
            )
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_aqi_visulisation(
                self.air_pollution_DF_8am, "invalid", "no2 AQI", "output"
            )
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.change_in_aqi_visulisation(
                self.air_pollution_DF_8am.drop(columns=["no2 AQI"]),
                self.air_pollution_DF_9am, "no2 AQI", "output"
            )

    @patch("environmental_insights.air_pollution_functions.change_in_aqi_visulisation")
    def test_change_in_aqi_visulisation_correct(self, mock_func):
        ei_air_pollution_functions.change_in_aqi_visulisation(
            self.air_pollution_DF_8am, self.air_pollution_DF_9am, "no2 AQI", "output"
        )
        mock_func.assert_called_once()
