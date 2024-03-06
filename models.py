import pickle
import os
import pandas as pd
import variables
import numpy as np

def download_file_model(filename):
    """
    Checks if a file that has been requested has been downloaded and if it has then it will download the file

    Parameters:
    filename (string): The dataset filename to be downloaded from the remote server
    """
    print("Checking existence of file: " + str(filename))

def load_model_united_kingdom(model_type, model_dataset, air_pollutant):
    """
    Load in a pre trained air pollution machine learning model for the UK.

    Parameters:
    model_type (string): The model type to load in for the quantile regression, options: 0.95, 0.5, 0.05.
    model_dataset (string): The underpinning dataset that the model has been trained on
    air_pollutant (string):  The air pollutant the model has been trained on.

    Return:
    lightGBMRegression models: A lightGBM instance of the model.
    """
    model_filepath = "environmental_insights_models/uk/dataset_"+model_dataset+"_quantile_regression_"+model_type+"_air_pollutant_"+air_pollutant+".pkl"
    with open(model_filepath, "rb") as f:  # Python 3: open(..., 'rb')
        bootstrapModel = pickle.load(f)
    return bootstrapModel

def load_model_global(model_type, model_dataset, air_pollutant):
    """
    Load in a pre trained air pollution machine learning model for the globe.

    Parameters:
    model_type (string): The model type to load in for the quantile regression, options: 0.95, 0.5, 0.05.
    model_dataset (string): The underpinning dataset that the model has been trained on
    air_pollutant (string):  The air pollutant the model has been trained on.

    Return:
    lightGBMRegression models: A lightGBM instance of the model.
    """

def load_feature_vector_typical_day_united_kingdom(month, day_of_week, hour, uk_grids):
    """
    Load in a feature vector for the typical day in the United Kingdom.

    Parameters:
    month (int): An int to represent the month of interest, 1 being january, and 12 being December.
    day_of_week (string): A string to represent the day of week of interest in the form of "Friday".
    hour (int):  An int to represent the hour of interest, 0 being midnight, and 23 being the final possible hour of the day
    uk_grids (geodataframe): A Geodataframe that describes the estimation points for the uk model

    Returns:
    geodataframe: A geodataframe of the typical dataset feature vector in the UK.
    """
    desire_filename = "environmental_insights_data/feature_vector/uk_typical_day/Month_"+str(month)+"-Day_"+day_of_week+"-Hour_"+str(hour)+".feather"
    if not os.path.isfile(desire_filename):
        download_file_model(desire_filename)

    feature_vector = pd.read_feather(desire_filename)
    feature_vector = feature_vector.rename(columns={"Grid ID":"UK Model Grid ID"})
    feature_vector = uk_grids.merge(feature_vector, on="UK Model Grid ID")
    return feature_vector

def get_model_feature_vector(model_type):
    """
    Getter function to return the list of features that were used for a given model type.

    Parameters:
    model_type (string): Whether the model requested is the one with all of the features or just the transport infrastructure features.

    Returns:
    list: A list of the feature vector names that were used in the model request.
    """
    return variables.featureVectorSubsets[model_type]

def make_concentration_predicitions_united_kingdom(estimating_model, observation_data, estimating_feature_vector_column_names):
    """
    Make predicition for a given environment conditions for air pollution concentrations.

    Parameters:
    estimating_model (string): Whether the model requested is the one with all of the features or just the transport infrastructure features.
    observation_data (dataframe): The observational data for the environmental conditions to make the predicitons on.
    estimating_feature_vector_column_names (list): A list of the feature vector names that were used in the model request.

    Returns:
    geoodataframe: Predictions for the given air pollutant for the given environmental conditions.
    """
    feature_vector_DF = observation_data.copy(deep=True)

    timestamps = feature_vector_DF.index



    in_seqs = list()
    for columnName in estimating_feature_vector_column_names:
        in_seq = feature_vector_DF[[columnName]].to_numpy()
        in_seq = in_seq.reshape((len(in_seq), 1))
        in_seqs.append(in_seq)

    feature_vector = np.hstack(tuple(in_seqs))
    feature_vector = feature_vector[:,:feature_vector.shape[1]]



    predcited_pollution_comparison = feature_vector_DF[["UK Model Grid ID"]].copy(deep=True)

    predicitionColumnNames = list()

    predcited_pollution = estimating_model.predict(feature_vector)
    predcited_pollution = np.exp(predcited_pollution) - 0.0000001
    predcited_pollution_comparison["Model Predicition"] = predcited_pollution
    return predcited_pollution_comparison
