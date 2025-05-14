import unittest
import os
import sys
import geopandas as gpd
import pandas as pd
import xarray as xr
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import environmental_insights.data as ei_data

script_dir = os.path.dirname(__file__)


class TestAirPollutionDataFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.uk_grids = ei_data.get_uk_grids()

    def test_read_nc_correct_input(self):
        test_file = os.path.join(script_dir, "..", "test_data", "small_test.nc")
        if os.path.exists(test_file):
            ds = ei_data.read_nc(test_file)
            self.assertIsInstance(ds, xr.Dataset)
        else:
            self.skipTest(f"Test file {test_file} not found")

    def test_read_nc_invalid_input(self):
        with self.assertRaises(FileNotFoundError):
            ei_data.read_nc("non_existent_file.nc")

    def test_netcdf_to_dataframe_correct_input(self):
        ds = xr.Dataset({"dummy_var": (("x",), [1, 2, 3])})
        df = ei_data.netcdf_to_dataframe(ds)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("dummy_var", df.columns)

    def test_air_pollution_concentration_typical_day_real_time_united_kingdom_correct(self):
        df = ei_data.air_pollution_concentration_typical_day_real_time_united_kingdom(1, "Monday", 8)
        self.assertIsInstance(df, pd.DataFrame)

    def test_air_pollution_concentration_typical_day_real_time_united_kingdom_invalid_type(self):
        with self.assertRaises(ValueError):
            ei_data.air_pollution_concentration_typical_day_real_time_united_kingdom(1, "Monday", 8, "WrongType")

    def test_air_pollution_concentration_nearest_point_typical_day_united_kingdom_correct(self):
        df = ei_data.air_pollution_concentration_nearest_point_typical_day_united_kingdom(1, "Monday", 8, 51.5, 0.12, self.uk_grids)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    def test_air_pollution_concentration_nearest_point_typical_day_united_kingdom_invalid_type(self):
        with self.assertRaises(ValueError):
            ei_data.air_pollution_concentration_nearest_point_typical_day_united_kingdom(1, "Monday", 8, 51.5, 0.12, self.uk_grids, "WrongType")

    def test_air_pollution_concentration_complete_set_real_time_united_kingdom_correct(self):
        df = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom("2018-01-01_080000")
        self.assertIsInstance(df, pd.DataFrame)

    def test_air_pollution_concentration_complete_set_real_time_united_kingdom_invalid_type(self):
        with self.assertRaises(ValueError):
            ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom("2018-01-01_080000", "WrongType")

    def test_get_uk_grids(self):
        gdf = ei_data.get_uk_grids()
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertIn("UK_Model_Grid_ID", gdf.columns)

    def test_get_global_grids(self):
        gdf = ei_data.get_global_grids()
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertIn("Global Model Grid ID", gdf.columns)

    @patch("environmental_insights.data.overpy.Overpass.query")
    def test_get_amenities_as_geodataframe_mock(self, mock_query):
        mock_result = MagicMock()
        mock_result.nodes = []
        mock_result.ways = []
        mock_result.relations = []
        mock_query.return_value = mock_result

        gdf = ei_data.get_amenities_as_geodataframe("hospital", 51.5, -0.1, 51.6, 0.1)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)

    @patch("environmental_insights.data.overpy.Overpass.query")
    def test_get_highways_as_geodataframe_mock(self, mock_query):
        mock_result = MagicMock()
        mock_result.ways = []
        mock_query.return_value = mock_result

        gdf = ei_data.get_highways_as_geodataframe("residential", 51.5, -0.1, 51.6, 0.1)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)

    def test_replace_feature_vector_column(self):
        original_df = pd.DataFrame({
            "UK_Model_Grid_ID": [1, 2, 3],
            "Feature1": [10, 20, 30]
        })
        new_df = pd.DataFrame({
            "UK_Model_Grid_ID": [1, 2, 3],
            "Feature1": [100, 200, 300]
        })
        updated_df = ei_data.replace_feature_vector_column(original_df, new_df, "Feature1")
        self.assertTrue((updated_df["Feature1"] == [100, 200, 300]).all())
