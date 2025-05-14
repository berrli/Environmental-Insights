import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import environmental_insights.data as ei_data
import environmental_insights.models as ei_models
import environmental_insights.air_pollution_functions as ei_air


class IntegrationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define parameters for the workflow
        cls.timestamp = "2018-01-01_080000"
        cls.pollutant = "no2"

        # Establish test directory paths
        cls.test_dir = Path(__file__).parent.resolve()
        cls.output_dir = cls.test_dir / "../visualisations"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

                # Monkey-patch the package's output directory to use the test directory (as Path)
        ei_air.VISUALISATIONS_DIR = cls.output_dir

        # Load UK model grids
        cls.uk_grids = ei_data.get_uk_grids()

        # Download and load input and output datasets for the given timestamp
        cls.df_input = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom(
            cls.timestamp, "Input"
        )
        cls.df_output = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom(
            cls.timestamp, "Output"
        )

        # Load trained model and feature names
        cls.model = ei_models.load_model_united_kingdom("mean", cls.pollutant, "All")
        cls.feature_names = ei_models.get_model_feature_vector("All")

        # Generate model predictions on the input data
        cls.predictions = ei_models.make_concentration_predictions_united_kingdom(
            cls.model, cls.df_input, cls.feature_names
        )

        # Convert the real output dataset to UK AQI
        cls.aqi_df = ei_air.air_pollution_concentrations_to_UK_daily_air_quality_index(
            cls.df_output,
            cls.pollutant,
            f"{cls.pollutant}_Prediction_Mean",
        )

        # Prepare merged geodataframes for visualisation
        cls.aqi_merged = cls.uk_grids.merge(cls.aqi_df, on="UK_Model_Grid_ID")

    def test_input_output_nonempty(self):
        """Input and output datasets should be non-empty after download and load."""
        self.assertFalse(self.df_input.empty, "Input dataset should not be empty.")
        self.assertFalse(self.df_output.empty, "Output dataset should not be empty.")

    def test_model_predictions(self):
        """Model predictions should include the 'Model Prediction' column and contain data."""
        self.assertIn("Model Prediction", self.predictions.columns)
        self.assertGreater(len(self.predictions), 0, "Predictions DataFrame should contain rows.")

    def test_aqi_conversion(self):
        """AQI conversion should add the correct AQI column and return data."""
        aqi_col = f"{self.pollutant} AQI"
        self.assertIn(aqi_col, self.aqi_df.columns)
        self.assertGreater(len(self.aqi_df), 0, "AQI DataFrame should contain rows.")

    def test_nearest_point(self):
        """Retrieving the nearest point data should return a single point with distance measurement."""
        nearest = ei_data.air_pollution_concentration_nearest_point_real_time_united_kingdom(
            51.5, 0.12, self.timestamp, self.uk_grids
        )
        # Expect a single closest grid point
        self.assertEqual(nearest.shape[0], 1, "Nearest-point result should have exactly one row.")
        self.assertIn("Prediction Latitude", nearest.columns)
        self.assertIn("Distance", nearest.columns)

    def test_visualisations_are_saved(self):
        """Visualisation functions should complete without error and save PNG files."""
        out_dir = Path(ei_air.VISUALISATIONS_DIR)

        # AQI map
        aqi_map_filename = f"test_{self.pollutant}_aqi"
        ei_air.visualise_air_pollution_daily_air_quality_index(
            self.aqi_merged,
            f"{self.pollutant} AQI",
            aqi_map_filename,
        )
        self.assertTrue(
            (out_dir / f"{aqi_map_filename}.png").exists(),
            f"{aqi_map_filename}.png should be created in the visualisations directory",
        )

        # Change in concentration map between 8am and 9am
        df_9am = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom(
            "2018-01-01_090000", "Output"
        )
        aqi_9am = ei_air.air_pollution_concentrations_to_UK_daily_air_quality_index(
            df_9am,
            self.pollutant,
            f"{self.pollutant}_Prediction_Mean",
        )
        merged_9am = self.uk_grids.merge(aqi_9am, on="UK_Model_Grid_ID")

        conc_change_filename = "test_conc_change"
        ei_air.change_in_concentrations_visulisation(
            self.aqi_merged,
            merged_9am,
            f"{self.pollutant}_Prediction_Mean",
            conc_change_filename,
        )
        self.assertTrue(
            (out_dir / f"{conc_change_filename}.png").exists(),
            f"{conc_change_filename}.png should be created",
        )

        # Change in AQI map between 8am and 9am
        aqi_change_filename = "test_aqi_change"
        ei_air.change_in_aqi_visulisation(
            self.aqi_merged,
            merged_9am,
            f"{self.pollutant} AQI",
            aqi_change_filename,
        )
        self.assertTrue(
            (out_dir / f"{aqi_change_filename}.png").exists(),
            f"{aqi_change_filename}.png should be created",
        )