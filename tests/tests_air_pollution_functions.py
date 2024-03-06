import unittest
import sys
sys.path.append("../") # Adds higher directory to python modules path.
import air_pollution_functions as ei_air_pollution_functions
import data as ei_data



class air_pollution_functions_arguement_defence(unittest.TestCase):

    ######
    ###### Needed data variables for the software. These have been manually tested so far due to the data uploading process. 
    ######
    uk_grids = ei_data.get_uk_grids()
    global_grids = ei_data.get_global_grids()
    uk_complete_dataset = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom("2018-01-01 080000")
    uk_single_datapoint = ei_data.air_pollution_concentration_nearest_point_real_time_united_kingdom(51.5, 0.12, "2018-01-01 080000", uk_grids)
    
    global_complete_dataset = ei_data.air_pollution_concentration_complete_set_real_time_global("07-02-2022 080000")
    air_pollution_DF_daily_air_quality_index_global = ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(global_complete_dataset, "no2", "no2")
    air_pollution_DF_daily_air_quality_index_global = global_grids.merge(air_pollution_DF_daily_air_quality_index_global, on="Global Model Grid ID")
    
    air_pollution_DF_8am = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom("2018-01-01 080000")
    air_pollution_DF_9am = ei_data.air_pollution_concentration_complete_set_real_time_united_kingdom("2018-01-01 090000")
    air_pollution_DF_8am = ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(air_pollution_DF_8am, "no2", "no2 Prediction mean")
    air_pollution_DF_9am = ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(air_pollution_DF_9am, "no2", "no2 Prediction mean")
    
    air_pollution_DF_8am = uk_grids.merge(air_pollution_DF_8am, on="UK Model Grid ID")
    air_pollution_DF_9am = uk_grids.merge(air_pollution_DF_9am, on="UK Model Grid ID")
    
    #Persistent variables throughout the tests
    air_pollutants = ["no2", "o3", "pm10", "pm2.5", "so2"]


    
    ####
    #### Tests for the function "air_pollution_concentrations_to_UK_daily_air_quality_index"
    ####
    
    def test_air_pollution_concentrations_to_UK_daily_air_quality_index_correct_input(self):
        #Correct input for each of the different air pollutant
        for air_pollutant in self.air_pollutants:
            ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(self.uk_complete_dataset, air_pollutant, air_pollutant + " Prediction mean")

    def test_air_pollution_concentrations_to_UK_daily_air_quality_index_incorrect_input(self):
        #Incorrect air pollutant name
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(self.uk_complete_dataset, "nonsense", "no2 Prediction mean")

        #incorrect dataframe passed, string rather than dataframe
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index("Dataframe", "no2", "no2 Prediction mean")

        #incorrect column name, the column name is not within the dataframe.
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index(self.uk_complete_dataset, "no2", "This isnt in the dataframe columns!")

    ####
    #### Tests for the function "visualise_air_pollution_daily_air_quality_index"
    ####
    def test_visualise_air_pollution_daily_air_quality_index_correct_input(self):
        ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_index(self.air_pollution_DF_daily_air_quality_index_global, "no2 AQI", "uk_2018_01_01_080000_air_quality_index")
    
    def test_visualise_air_pollution_daily_air_quality_index_incorrect_input(self):
        #incorrect aqi name, not a valid aqi format. 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_index(self.air_pollution_DF_daily_air_quality_index_global, "Incorrect Output", "uk_2018_01_01_080000_air_quality_index")

        #Incorrect  AQI value, no is not one of the five air pollutants that makes up the DAQI 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_index(self.air_pollution_DF_daily_air_quality_index_global, "no AQI", "uk_2018_01_01_080000_air_quality_index")

        #Not a geodataframe that has been passed. 
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_index("geodataframe", "no2 AQI", "uk_2018_01_01_080000_air_quality_index")

        #The AQI to be plotted is not wihtin the dataframe. 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_index(self.air_pollution_DF_daily_air_quality_index_global.drop(columns=["no2 AQI"]), "no2 AQI", "uk_2018_01_01_080000_air_quality_index")

        
    ####
    #### Tests for the function "visualise_air_pollution_daily_air_quality_bands"
    ####

    def test_visualise_air_pollution_daily_air_quality_bands_correct_input(self):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands(self.air_pollution_DF_daily_air_quality_index_global, "no2 Air Quality Index AQI Band", "test")
        #for air_pollutant in self.air_pollutants:
        #    ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands(self.air_pollution_DF_daily_air_quality_index_global, air_pollutant+ " Air Quality Index AQI Band", "test")
        
    def test_visualise_air_pollution_daily_air_quality_bands_incorrect_input(self):
        #incorrect aqi name, not a valid aqi format. 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands(self.air_pollution_DF_daily_air_quality_index_global, "Incorrect Output", "test")

        #Incorrect  AQI value, no is not one of the five air pollutants that makes up the DAQI 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands(self.air_pollution_DF_daily_air_quality_index_global, "no Air Quality Index AQI Band", " test")

         #Not a geodataframe that has been passed. 
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands("geodataframe", "no2 Air Quality Index AQI Band", "test")

        #The AQI to be plotted is not wihtin the dataframe. 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.visualise_air_pollution_daily_air_quality_bands(self.air_pollution_DF_daily_air_quality_index_global.drop(columns=["no2 Air Quality Index AQI Band"]), "no2 Air Quality Index AQI Band", "uk_2018_01_01_080000_air_quality_index")

            
    ####
    #### Tests for the function "change_in_concentrations_visulisation"
    ####

    def test_change_in_concentrations_visulisation_correct_input(self):
        for air_pollutant in self.air_pollutants: 
            ei_air_pollution_functions.change_in_concentrations_visulisation(self.air_pollution_DF_8am, self.air_pollution_DF_9am, air_pollutant + " Prediction mean", "uk_concentration_change_between_8_9_am")



    def test_change_in_concentrations_visulisation_incorrect_input(self):

        #Ensure that the first dataframe that is passed is a dataframe. 
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_concentrations_visulisation("dataframe", self.air_pollution_DF_9am, "no2 Prediction mean", "uk_concentration_change_between_8_9_am")

        #Ensure that the second dataframe that is passed is a dataframe. 
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_concentrations_visulisation(self.air_pollution_DF_8am, "dataframe", "no2 Prediction mean", "uk_concentration_change_between_8_9_am")

        #Ensure that the first dataframe has the relevenat column name needed.
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.change_in_concentrations_visulisation(self.air_pollution_DF_8am.drop(columns=["no2 Prediction mean"]), self.air_pollution_DF_9am, "no2 Prediction mean", "uk_concentration_change_between_8_9_am")

        #Ensure that the second dataframe has the releveant column name needed. 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.change_in_concentrations_visulisation(self.air_pollution_DF_8am, self.air_pollution_DF_9am.drop(columns=["no2 Prediction mean"]), "no2 Prediction mean", "uk_concentration_change_between_8_9_am")
    ####
    #### Tests for the function "change_in_aqi_visulisation"
    ####

    def test_change_in_aqi_visulisation_correct_input(self):
        ei_air_pollution_functions.change_in_aqi_visulisation(self.air_pollution_DF_8am, self.air_pollution_DF_9am, "no2 AQI",  "uk_aqi_change_between_8_9_am")

    def test_change_in_aqi_visulisation_incorrect_input(self):

        #Ensure that the first dataframe that is passed is a dataframe. 
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_aqi_visulisation("dataframe", self.air_pollution_DF_9am, "no2 AQI",  "uk_aqi_change_between_8_9_am")

        #Ensure that the second dataframe that is passed is a dataframe. 
        with self.assertRaises(TypeError):
            ei_air_pollution_functions.change_in_aqi_visulisation(self.air_pollution_DF_8am, "dataframe", "no2 AQI",  "uk_aqi_change_between_8_9_am")

        #Ensure that the first dataframe has the relevenat column name needed.
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.change_in_aqi_visulisation(self.air_pollution_DF_8am.drop(columns=["no2 AQI"]), self.air_pollution_DF_9am, "no2 AQI",  "uk_aqi_change_between_8_9_am")

        #Ensure that the second dataframe has the releveant column name needed. 
        with self.assertRaises(ValueError):
            ei_air_pollution_functions.change_in_aqi_visulisation(self.air_pollution_DF_8am, self.air_pollution_DF_9am.drop(columns=["no2 AQI"]), "no2 AQI",  "uk_aqi_change_between_8_9_am")


    ####
    #### Tests for the function "change_in_concentration_line"
    ####

    # TO DO once the final version of the data has been completed. 