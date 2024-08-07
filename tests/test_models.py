import unittest
import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../environmental_insights')))

import sys
import os

# Calculate the absolute path to the module's directory
module_path = os.path.abspath(os.path.join('..', 'environmental_insights'))

# Add this directory to the system path
sys.path.insert(0, module_path) 

import environmental_insights.air_pollution_functions as ei_air_pollution_functions
import environmental_insights.data as ei_data
import environmental_insights.models as ei_models


class models_arguement_defence(unittest.TestCase):
    def test_placeholders(self):
        pass

    # To do once the final location of the data has been decided.
