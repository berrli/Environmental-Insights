import math
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
import environmental_insights.variables as variables  # Absolute import
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'

# Get the root directory of the package
PACKAGE_ROOT = Path(__file__).resolve().parent
VISUALISATIONS_DIR = (
    PACKAGE_ROOT / "environmental_insights/environmental_insights_visulisations"
)

# Ensure the directory exists
VISUALISATIONS_DIR.mkdir(parents=True, exist_ok=True)


def air_pollution_concentrations_to_UK_daily_air_quality_index(
    predicitions, pollutant, air_pollutant_column_name
):
    """
    Add onto an existing dataframe the Daily Air Quality Index (https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
    for the air pollutant concentration data described.

    Parameters:
    predicitions (dataframe): A dataframe of the air pollution concentrations that are to be added onto
    pollutant (string): The string of the air pollutant concentration thresholds to be used to create the air quality indexes.
    air_pollutant_column_name (string): The string of the column name for the air pollution concentration to calculate the air quality index on.

    Returns:
    dataframe: A dataframe with the additional columns for the air quality index based on the outlined air pollution concentration data.
    """

    #####
    ##### Defensive Programming
    #####

    # Check the predicitions is a dataframe
    if not isinstance(predicitions, pd.DataFrame):
        raise TypeError(
            "Please ensure that the datatype for predicitions is a dataframe, the recieved arguement is: "
            + str(type(predicitions))
        )

    # Check a valid air pollutant has been inputted.
    if pollutant not in ["o3", "no2", "nox", "no", "so2", "pm2p5", "pm10"]:
        raise ValueError(
            'Please ensure that the value for pollutant is one of ["o3", "no2", "so2", "pm2p5", "pm10"], the recieved arguement is: '
            + pollutant
            + " of the type: "
            + str(type(pollutant))
        )

    # Ensure that the air pollutant that is to be ran is actually within the dataframe that was passed.
    if air_pollutant_column_name not in predicitions.columns:
        raise ValueError(
            "Please ensure that the air_pollutant_column_name passed is within the dataframe predicitions, it is currently not."
        )

    #####

    air_quality_index_bins = dict()
    air_quality_index_bins["o3"] = [
        0,
        33.5,
        66.5,
        100.5,
        120.5,
        140.5,
        160.5,
        187.5,
        213.5,
        240.5,
        math.inf,
    ]
    air_quality_index_bins["no2"] = [
        0,
        67.5,
        134.5,
        200.5,
        267.5,
        334.5,
        400.5,
        467.5,
        534.5,
        600.5,
        math.inf,
    ]
    air_quality_index_bins["so2"] = [
        0,
        88.5,
        177.5,
        266.5,
        354.5,
        443.5,
        532.5,
        710.5,
        887.5,
        1064.5,
        math.inf,
    ]
    air_quality_index_bins["pm25"] = [
        0,
        11.5,
        23.5,
        35.5,
        41.5,
        47.5,
        53.5,
        58.5,
        64.5,
        70.5,
        math.inf,
    ]
    air_quality_index_bins["pm10"] = [
        0,
        16.5,
        33.5,
        50.5,
        58.5,
        66.5,
        75.5,
        83.5,
        91.5,
        100.5,
        math.inf,
    ]

    air_quality_index_bins = dict()
    if pollutant == "o3":
        air_quality_index_bins["o3"] = [
            0,
            33.5,
            66.5,
            100.5,
            120.5,
            140.5,
            160.5,
            187.5,
            213.5,
            240.5,
            math.inf,
        ]
        predicitions["o3 AQI"] = pd.cut(
            predicitions[air_pollutant_column_name],
            right=False,
            bins=air_quality_index_bins["o3"],
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
    elif pollutant == "no2":
        air_quality_index_bins["no2"] = [
            0,
            67.5,
            134.5,
            200.5,
            267.5,
            334.5,
            400.5,
            467.5,
            534.5,
            600.5,
            math.inf,
        ]
        predicitions["no2 AQI"] = pd.cut(
            predicitions[air_pollutant_column_name],
            right=False,
            bins=air_quality_index_bins["no2"],
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
    elif pollutant == "so2":
        air_quality_index_bins["so2"] = [
            0,
            88.5,
            177.5,
            266.5,
            354.5,
            443.5,
            532.5,
            710.5,
            887.5,
            1064.5,
            math.inf,
        ]
        predicitions["so2 AQI"] = pd.cut(
            predicitions[air_pollutant_column_name],
            right=False,
            bins=air_quality_index_bins["so2"],
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
    elif pollutant == "pm2p5":
        air_quality_index_bins["pm2p5"] = [
            0,
            11.5,
            23.5,
            35.5,
            41.5,
            47.5,
            53.5,
            58.5,
            64.5,
            70.5,
            math.inf,
        ]
        predicitions["pm2p5 AQI"] = pd.cut(
            predicitions[air_pollutant_column_name],
            right=False,
            bins=air_quality_index_bins["pm2p5"],
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
    elif pollutant == "pm10":
        air_quality_index_bins["pm10"] = [
            0,
            16.5,
            33.5,
            50.5,
            58.5,
            66.5,
            75.5,
            83.5,
            91.5,
            100.5,
            math.inf,
        ]
        predicitions["pm10 AQI"] = pd.cut(
            predicitions[air_pollutant_column_name],
            right=False,
            bins=air_quality_index_bins["pm10"],
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

    air_pollution_level_map = {
        1: "Low",
        2: "Low",
        3: "Low",
        4: "Moderate",
        5: "Moderate",
        6: "Moderate",
        7: "High",
        8: "High",
        9: "High",
        10: "Very High",
    }
    predicitions[pollutant + " Air Quality Index AQI Band"] = predicitions[
        pollutant + " AQI"
    ].map(air_pollution_level_map)
    return predicitions


def visualise_air_pollution_daily_air_quality_index(
    air_pollution_GDF, aqi_to_plot, filename
):
    """
    Visualise air_pollution_GDF with the UK Daily Air Quality Index (https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
    using the individual index bounds and standard color codes.

    Parameters:
    air_pollution_GDF (dataframe): A dataframe of the air pollution concentrations that are to be added onto
    aqi_to_plot (string): Name of the column within air_pollution_GDF that has the indexes that are to be plotted.
    filename (string): Filename for the visualisation that is outputted in the environmental_insights_visulisations directory
    """

    #####
    ##### Defensive Programming
    #####

    # Ensure that the format for the aqi_to_plot is of the correct format.
    if aqi_to_plot not in ["no2 AQI", "o3 AQI", "pm10 AQI", "pm2p5 AQI", "so2 AQI"]:
        raise ValueError(
            "Please ensure that the AQI to plot is of the correct form, namely one of the following: no2 AQI, o3 AQI, pm10 AQI, pm2p5 AQI, so2 AQI"
        )

    # Ensure that the dataframe passed is a geodataframe
    if not isinstance(air_pollution_GDF, gpd.GeoDataFrame):
        raise TypeError(
            "Please ensure that the datatype for predicitions is a dataframe, the recieved arguement is: "
            + str(type(air_pollution_GDF))
        )

    # Ensure that the air pollutant (aqi_to_plot) that is to be ran is actually within the geodataframe that was passed.
    if aqi_to_plot not in air_pollution_GDF.columns:
        raise ValueError(
            "Please ensure that the aqi_to_plot passed is within the geodataframe air_pollution_GDF, it is currently not."
        )

    #####

    # Plot the AQI
    colormap = {
        1: "#9cff9c",
        2: "#31ff00",
        3: "#31cf00",
        4: "#ff0",
        5: "#ffcf00",
        6: "#ff9a00",
        7: "#ff6464",
        8: "red",
        9: "#900",
        10: "#ce30ff",
    }
    fig, axes = plt.subplots(1, figsize=(10, 15))
    plt.subplots_adjust(wspace=0, hspace=0)

    for aqi_index in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        predictedPollutionSubset = air_pollution_GDF[
            air_pollution_GDF[aqi_to_plot] == int(aqi_index)
        ]
        if predictedPollutionSubset.shape[0] > 0:
            # predictedPollutionSubset.plot(ax=axes, column=aqi_to_plot, color=colormap[int(aqi_index)])
            predictedPollutionSubset.plot(ax=axes, color=colormap[int(aqi_index)])
    axes.axis("off")
    legend_items = list()
    for aqi_index in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        patch = mpatches.Patch(color=colormap[int(aqi_index)], label=aqi_index)

        legend_items.append(patch)

    plt.legend(handles=legend_items, title="UK DAQI")
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")


def visualise_air_pollution_daily_air_quality_bands(
    air_pollution_GDF, aqi_to_plot, filename
):
    """
    Visualise air_pollution_GDF with the UK Daily Air Quality Index (https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
    using the bands and standard color codes.

    Parameters:
    air_pollution_GDF (dataframe): A dataframe of the air pollution concentrations that are to be added onto
    aqi_to_plot (string): Name of the column within air_pollution_GDF that has the bands that are to be plotted.
    filename (string): Filename for the visualisation that is outputted in the environmental_insights_visulisations directory
    """

    #####
    ##### Defensive Programming
    #####

    # Ensure that the format for the aqi_to_plot is of the correct format.
    if aqi_to_plot not in [
        "no2 Air Quality Index AQI Band",
        "o3 Air Quality Index AQI Band",
        "pm10 Air Quality Index AQI Band",
        "pm2p5 Air Quality Index AQI Band",
        "so2 Air Quality Index AQI Band",
    ]:
        raise ValueError(
            "Please ensure that the AQI to plot is of the correct form, namely one of the following: no2 Air Quality Index AQI Band, o3 Air Quality Index AQI Band, pm10 Air Quality Index AQI Band, pm2p5 Air Quality Index AQI Band, so2 Air Quality Index AQI Band"
        )

    # Ensure that the dataframe passed is a geodataframe
    if not isinstance(air_pollution_GDF, gpd.GeoDataFrame):
        raise TypeError(
            "Please ensure that the datatype for air_pollution_GDF is a geodataframe, the recieved arguement is: "
            + str(type(air_pollution_GDF))
        )

    # Ensure that the air pollutant (aqi_to_plot) that is to be ran is actually within the geodataframe that was passed.
    if aqi_to_plot not in air_pollution_GDF.columns:
        raise ValueError(
            "Please ensure that the aqi_to_plot passed is within the geodataframe air_pollution_GDF, it is currently not."
        )

    #####

    fig, axes = plt.subplots(1, figsize=(10, 15))
    colormap = {
        "Low": "#31ff00",
        "Moderate": "#ffcf00",
        "High": "red",
        "Very High": "#ce30ff",
    }
    for aqi_index in ["Low", "Moderate", "High", "Very High"]:
        predictedPollutionSubset = air_pollution_GDF[
            air_pollution_GDF[aqi_to_plot] == aqi_index
        ]
        if predictedPollutionSubset.shape[0] > 0:
            # predictedPollutionSubset.plot(ax=axes, column=aqi_to_plot, color=colormap[aqi_index]
            predictedPollutionSubset.plot(ax=axes, color=colormap[aqi_index])
    axes.axis("off")
    legend_items = list()
    for aqi_index in ["Low", "Moderate", "High", "Very High"]:
        patch = mpatches.Patch(color=colormap[aqi_index], label=aqi_index)

        legend_items.append(patch)

    plt.legend(handles=legend_items, title="UK DAQI \n Bands")
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")


def change_in_concentrations_visulisation(
    first_dataframe, second_dataframe, air_pollutant, filename
):
    """
    Visualisation the change in concentrations for two datasets of air pollution concentrations based on actual concentrations.

    Parameters:
    Parameters:
    first_dataframe (DataFrame): The first concentration dataset.
    second_dataframe (DataFrame): The second concentration dataset.
    air_pollutant (string):  Common column name in both dataframes that will be used to calculate the difference in concentrations.
    filename (string): Filename for the visualisation that is outputted in the environmental_insights_visulisations directory
    """

    #####
    ##### Defensive Programming
    #####

    # Ensure that the fiurst dataframe passed is a geodataframe
    if not isinstance(first_dataframe, pd.DataFrame):
        raise TypeError(
            "Please ensure that the datatype for first_dataframe is a dataframe, the recieved arguement is: "
            + str(type(first_dataframe))
        )

    # Ensure that the second dataframe passed is a geodataframe
    if not isinstance(second_dataframe, pd.DataFrame):
        raise TypeError(
            "Please ensure that the datatype for second_dataframe is a dataframe, the recieved arguement is: "
            + str(type(second_dataframe))
        )

    # Ensure that the air_pollutant is in the columns for both dataframes.
    if air_pollutant not in first_dataframe.columns:
        raise ValueError(
            "Please ensure that the value for air pollutant is within first_dataframe."
        )
    if air_pollutant not in second_dataframe.columns:
        raise ValueError(
            "Please ensure that the value for air pollutant is within second_dataframe."
        )

    #####

    air_pollution_DF_change = first_dataframe[["geometry"]].copy(deep=True)
    air_pollution_DF_change[air_pollutant + " Change"] = (
        first_dataframe[air_pollutant] - second_dataframe[air_pollutant]
    )

    overall_min = air_pollution_DF_change[air_pollutant + " Change"].min()
    centerValue = 0
    overall_max = air_pollution_DF_change[air_pollutant + " Change"].max()

    divnorm = colors.TwoSlopeNorm(
        vmin=overall_min, vcenter=centerValue, vmax=overall_max
    )
    fig, axes = plt.subplots(1, figsize=(15, 15))
    axes.axis("off")
    air_pollution_DF_change.plot(
        ax=axes,
        column=air_pollutant + " Change",
        vmin=overall_min,
        vmax=overall_max,
        cmap="bwr",
        norm=divnorm,
        legend=True,
        legend_kwds={
            "format": "%.0f",
            "shrink": 0.5,
            "label": "Concentration Change (μg/m$^3$)",
        },
    )
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")


def change_in_aqi_visulisation(
    first_dataframe, second_dataframe, air_pollutant, filename
):
    """
    Visualisation the change in concentrations for two datasets of air pollution concentrations based on air quality indexes.

    Parameters:
    Parameters:
    first_dataframe (DataFrame): The first concentration dataset.
    second_dataframe (DataFrame): The second concentration dataset.
    air_pollutant (string):  Common column name in both dataframes that will be used to calculate the difference in concentrations.
    filename (string): Filename for the visualisation that is outputted in the environmental_insights_visulisations directory
    """

    #####
    ##### Defensive Programming
    #####

    # Ensure that the fiurst dataframe passed is a geodataframe
    if not isinstance(first_dataframe, pd.DataFrame):
        raise TypeError(
            "Please ensure that the datatype for first_dataframe is a dataframe, the recieved arguement is: "
            + str(type(first_dataframe))
        )

    # Ensure that the second dataframe passed is a geodataframe
    if not isinstance(second_dataframe, pd.DataFrame):
        raise TypeError(
            "Please ensure that the datatype for second_dataframe is a dataframe, the recieved arguement is: "
            + str(type(second_dataframe))
        )

    # Ensure that the air_pollutant is in the columns for both dataframes.
    if air_pollutant not in first_dataframe.columns:
        raise ValueError(
            "Please ensure that the value for air pollutant is within first_dataframe."
        )
    if air_pollutant not in second_dataframe.columns:
        raise ValueError(
            "Please ensure that the value for air pollutant is within second_dataframe."
        )

    #####

    air_pollution_DF_change = first_dataframe[["geometry"]].copy(deep=True)
    first_dataframe[air_pollutant] = first_dataframe[air_pollutant].astype(int)
    second_dataframe[air_pollutant] = second_dataframe[air_pollutant].astype(int)
    air_pollution_DF_change[air_pollutant + " Change"] = (
        first_dataframe[air_pollutant] - second_dataframe[air_pollutant]
    )

    overall_min = air_pollution_DF_change[air_pollutant + " Change"].min()
    centerValue = 0
    overall_max = air_pollution_DF_change[air_pollutant + " Change"].max()

    if overall_min > -1:
        overall_min = -1
    if overall_max < 1:
        overall_max = 1

    divnorm = colors.TwoSlopeNorm(
        vmin=overall_min, vcenter=centerValue, vmax=overall_max
    )
    fig, axes = plt.subplots(1, figsize=(15, 15))
    axes.axis("off")
    air_pollution_DF_change.plot(
        ax=axes,
        column=air_pollutant + " Change",
        vmin=overall_min,
        vmax=overall_max,
        cmap="bwr",
        norm=divnorm,
        legend=True,
        legend_kwds={"format": "%.0f", "shrink": 0.5, "label": "AQI Change (μg/m$^3$)"},
    )
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")


def change_in_concentration_line(
    air_pollutant, baseline_list, change_list, days, hours_covered, filename
):
    """
    Visualisation the change in concentrations for two datasets of air pollution concentrations in a line graph.

    Parameters:
    Parameters:
    air_pollutant (string): The name of the air pollutant to plot,
    baseline_list (list): List of the air pollution concentrations for the baseline scenario.
    change_list (list): List of the air pollution concentrations for the future scenario.
    days (list): The days the lists covers.
    hours_covered (list): The house the list covers.
    filename (string): Filename for the visualisation that is outputted in the environmental_insights_visulisations directory

    """

    #####
    ##### Defensive Programming
    #####

    # TO DO

    #####

    lineStyle = {"Mean": "solid", "Max": "dotted", "Total": "dashed"}
    fig, axes = plt.subplots(1, figsize=(20, 8))

    hours = []
    baseline_pollution_data = []
    updated_pollution_data = []
    timestamp_labels = []
    hour_counter = 0
    for day_of_week in days:
        baseline_pollution_day = baseline_list[day_of_week]
        change_pollution_day = change_list[day_of_week]
        for hour in hours_covered:
            baseline_pollution_hour = baseline_pollution_day[hour]
            change_pollution_hour = change_pollution_day[hour]
            timestamp_labels.append(day_of_week + " " + str(hour))
            hours.append(hour_counter)

            hour_counter = hour_counter + 1
            baseline_pollution_data.append(
                baseline_pollution_hour["Model Prediction"].mean()
            )
            updated_pollution_data.append(
                change_pollution_hour["Model Prediction"].mean()
            )
    single_pollutant_data = pd.DataFrame.from_dict(
        {
            "Labels": timestamp_labels,
            "Hour": hours,
            "Baseline Pollution": baseline_pollution_data,
            "Update Pollution": updated_pollution_data,
        }
    )

    for i in np.arange(10):
        N = 1
        single_pollutant_data.index = single_pollutant_data.index * (N + 1)
        single_pollutant_data = single_pollutant_data.reindex(
            np.arange(single_pollutant_data.index.max() + N + 1)
        )
        single_pollutant_data = single_pollutant_data[:-1]

        single_pollutant_data["Hour"] = single_pollutant_data["Hour"].interpolate()
        single_pollutant_data["Baseline Pollution"] = single_pollutant_data[
            "Baseline Pollution"
        ].interpolate()
        single_pollutant_data["Update Pollution"] = single_pollutant_data[
            "Update Pollution"
        ].interpolate()

    single_pollutant_data["Difference"] = (
        single_pollutant_data["Update Pollution"]
        - single_pollutant_data["Baseline Pollution"]
    )

    preX = single_pollutant_data["Hour"].tolist()
    preY1 = single_pollutant_data["Baseline Pollution"].tolist()
    preY2 = single_pollutant_data["Update Pollution"].tolist()

    x = preX
    y1 = preY1
    y2 = preY2

    xx = np.repeat(x, 2)[1:]
    yy1 = np.repeat(y1, 2)[:-1]
    yy2 = np.repeat(y2, 2)[:-1]

    axes.fill_between(
        xx,
        yy1,
        yy2,
        color="#DC3220",
        where=yy1 < yy2,
        label=variables.replacePollutantName[air_pollutant] + " Increase",
        interpolate=True,
    )
    axes.fill_between(
        xx,
        yy1,
        yy2,
        color="#005AB5",
        where=yy1 > yy2,
        label=variables.replacePollutantName[air_pollutant] + " Decrease",
        interpolate=True,
    )

    # Potentially it is worth plotting just the baseline and seeing what that looks like
    axes.plot(
        single_pollutant_data["Hour"].tolist(),
        single_pollutant_data["Baseline Pollution"].tolist(),
        label=variables.replacePollutantName[air_pollutant] + " Baseline",
        color="black",
        linewidth=2.5,
        alpha=0.5,
    )

    axes.set_xlabel("Time", fontsize=15)
    axes.set_ylabel(
        variables.replacePollutantName[air_pollutant] + " Concentration (μg/m$^3$)",
        fontsize=15,
    )

    single_pollutant_data["Labels"] = single_pollutant_data["Labels"].fillna("")
    plt.xticks(
        single_pollutant_data["Hour"].tolist(),
        labels=single_pollutant_data["Labels"].tolist(),
    )
    axes.set_xlim(xx[0], xx[-1])
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )

    temp = axes.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.setp(
        axes.xaxis.get_majorticklabels(),
        rotation=60,
        fontsize=10,
        ha="right",
        rotation_mode="anchor",
    )
    axes.legend(loc="upper left")
    plt.show()
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")
