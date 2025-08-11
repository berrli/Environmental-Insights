import math
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from matplotlib import colors
import numpy as np
import environmental_insights.variables as variables  # Absolute import
from environmental_insights.data import get_uk_grids_outline
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'

# Get the root directory of the package
PACKAGE_ROOT = Path(__file__).resolve().parent
VISUALISATIONS_DIR = (
    PACKAGE_ROOT / "environmental_insights/environmental_insights_visulisations"
)
# Ensure the directory exists
VISUALISATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_gdf_and_column(gdf, column, allowed_values, name):
    """
    Internal helper to validate that `gdf` is a GeoDataFrame,
    that `column` is in its columns, and that it's one of the allowed values.
    """
    if column not in allowed_values:
        raise ValueError(f"{name} must be one of {allowed_values!r}, got {column!r}")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"{name!r} must be a GeoDataFrame, got {type(gdf)}")
    if column not in gdf.columns:
        raise ValueError(f"Column {column!r} not found in GeoDataFrame")


def _create_base_axes(plot_uk_outline, figsize):
    """
    Internal helper to create a matplotlib Axes,
    optionally drawing the UK grid outline beneath.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    if plot_uk_outline:
        uk_outline = get_uk_grids_outline()
        uk_outline.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=2,
            zorder=0,
        )
    ax.axis("off")
    return fig, ax


def _plot_layers(ax, gdf, column, colormap, zorder_start=1):
    """
    Internal helper to plot each unique value in `column` of `gdf`
    with the corresponding color in `colormap`. Returns legend patches.
    """
    legend_patches = []
    for i, (val, color) in enumerate(colormap.items(), start=zorder_start):
        subset = gdf[gdf[column] == val]
        if not subset.empty:
            subset.plot(ax=ax, color=color, zorder=i)
        legend_patches.append(mpatches.Patch(color=color, label=str(val)))
    return legend_patches


def _save_figure(fig, filename):
    """
    Internal helper to save and close a matplotlib Figure.
    """
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")
    if matplotlib.is_interactive():
        plt.show()
    plt.close(fig)


def air_pollution_concentrations_to_UK_daily_air_quality_index(
    predicitions, pollutant, air_pollutant_column_name
):
    """
    Add onto an existing dataframe the Daily Air Quality Index
    (https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
    for the air pollutant concentration data described.

    Parameters:
    -----------
    predicitions : pandas.DataFrame
        A dataframe of the air pollution concentrations that are to be added onto.
    pollutant : str
        The string of the air pollutant concentration thresholds to be used
        to create the air quality indexes.
    air_pollutant_column_name : str
        The string of the column name for the air pollution concentration
        to calculate the air quality index on.

    Returns:
    --------
    pandas.DataFrame
        A dataframe with the additional columns for the air quality index
        based on the outlined air pollution concentration data.
    """
    # Defensive programming checks
    if not isinstance(predicitions, pd.DataFrame):
        raise TypeError(
            "Please ensure that the datatype for predicitions is a dataframe, "
            f"received: {type(predicitions)}"
        )
    if pollutant not in ["o3", "no2", "nox", "no", "so2", "pm2p5", "pm10"]:
        raise ValueError(
            'Please ensure pollutant is one of ["o3", "no2", "nox", "no", "so2", "pm2p5", "pm10"], '
            f"received: {pollutant}"
        )
    if air_pollutant_column_name not in predicitions.columns:
        raise ValueError(
            f"Column {air_pollutant_column_name!r} not found in predictions DataFrame."
        )

    # Define the AQI breakpoints
    bins_map = {
        "o3":   [0, 33.5, 66.5, 100.5, 120.5, 140.5, 160.5, 187.5, 213.5, 240.5, math.inf],
        "no2":  [0, 67.5, 134.5, 200.5, 267.5, 334.5, 400.5, 467.5, 534.5, 600.5, math.inf],
        "so2":  [0, 88.5, 177.5, 266.5, 354.5, 443.5, 532.5, 710.5, 887.5, 1064.5, math.inf],
        "pm2p5":[0, 11.5, 23.5, 35.5, 41.5, 47.5, 53.5, 58.5, 64.5, 70.5, math.inf],
        "pm10": [0, 16.5, 33.5, 50.5, 58.5, 66.5, 75.5, 83.5, 91.5, 100.5, math.inf],
    }

    # Compute the AQI category
    labels = list(range(1, 11))
    predicitions[f"{pollutant} AQI"] = pd.cut(
        predicitions[air_pollutant_column_name],
        bins=bins_map[pollutant],
        right=False,
        labels=labels,
    )

    # Map AQI to band names
    air_pollution_level_map = {
        1: "Low", 2: "Low", 3: "Low",
        4: "Moderate", 5: "Moderate", 6: "Moderate",
        7: "High", 8: "High", 9: "High",
        10: "Very High",
    }
    predicitions[f"{pollutant} Air Quality Index AQI Band"] = (
        predicitions[f"{pollutant} AQI"].map(air_pollution_level_map)
    )

    return predicitions


def visualise_air_pollution_daily_air_quality_index(
    air_pollution_GDF, aqi_to_plot, filename, plot_uk_outline: bool = False,
):
    """
    Visualise air_pollution_GDF with the UK Daily Air Quality Index
    (https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
    using the individual index bounds and standard color codes.

    Parameters:
    -----------
    air_pollution_GDF : geopandas.GeoDataFrame
        A geodataframe of the air pollution concentrations that are to be plotted.
    aqi_to_plot : str
        Name of the column within air_pollution_GDF that has the indexes to plot.
        Must be one of ["no2 AQI", "o3 AQI", "pm10 AQI", "pm2p5 AQI", "so2 AQI"].
    filename : str
        Filename for the visualisation (PNG) in the visualisations directory.
    plot_uk_outline : bool, default False
        If True, draw the 1km UK grid outline first in a thin black line.
    """
    # Defensive programming
    allowed = ["no2 AQI", "o3 AQI", "pm10 AQI", "pm2p5 AQI", "so2 AQI"]
    _validate_gdf_and_column(air_pollution_GDF, aqi_to_plot, allowed, "aqi_to_plot")

    # Define the color map for AQI
    colormap = {
        1: "#9cff9c", 2: "#31ff00", 3: "#31cf00",
        4: "#ff0",    5: "#ffcf00", 6: "#ff9a00",
        7: "#ff6464", 8: "red",     9: "#900",
        10: "#ce30ff",
    }

    # Create base figure & axes, optionally with UK outline
    fig, ax = _create_base_axes(plot_uk_outline, figsize=(5, 5))

    # Plot each AQI category
    legend_patches = _plot_layers(ax, air_pollution_GDF, aqi_to_plot, colormap)

    # Add legend and save
    ax.legend(
        handles=legend_patches,
        title="UK DAQI",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.
    )
    _save_figure(fig, filename)


def visualise_air_pollution_daily_air_quality_bands(
    air_pollution_GDF, aqi_to_plot, filename, plot_uk_outline: bool = False,
):
    """
    Visualise air_pollution_GDF with the UK Daily Air Quality Index bands
    (https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
    using the band names and standard color codes.

    Parameters:
    -----------
    air_pollution_GDF : geopandas.GeoDataFrame
    aqi_to_plot : str
        Name of the column with AQI bands. Must be one of:
        ["no2 Air Quality Index AQI Band", "o3 Air Quality Index AQI Band",
         "pm10 Air Quality Index AQI Band", "pm2p5 Air Quality Index AQI Band",
         "so2 Air Quality Index AQI Band"].
    filename : str
    plot_uk_outline : bool, default False
    """
    allowed = [
        "no2 Air Quality Index AQI Band",
        "o3 Air Quality Index AQI Band",
        "pm10 Air Quality Index AQI Band",
        "pm2p5 Air Quality Index AQI Band",
        "so2 Air Quality Index AQI Band",
    ]
    _validate_gdf_and_column(air_pollution_GDF, aqi_to_plot, allowed, "aqi_to_plot")

    colormap = {
        "Low": "#31ff00",
        "Moderate": "#ffcf00",
        "High": "red",
        "Very High": "#ce30ff",
    }

    fig, ax = _create_base_axes(plot_uk_outline, figsize=(5, 5))
    legend_patches = _plot_layers(ax, air_pollution_GDF, aqi_to_plot, colormap)
    ax.legend(
        handles=legend_patches,
        title="UK DAQI Bands",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.
    )
    _save_figure(fig, filename)


def change_in_concentrations_visulisation(
    first_dataframe, second_dataframe, air_pollutant, filename, plot_uk_outline: bool = False,
):
    """
    Visualisation the change in concentrations for two datasets of air pollution
    concentrations based on actual concentrations.

    Parameters:
    -----------
    first_dataframe : pandas.DataFrame
        The first concentration dataset.
    second_dataframe : pandas.DataFrame
        The second concentration dataset.
    air_pollutant : str
        Common column name in both dataframes used to calculate the difference.
    filename : str
        Filename for the output PNG.
    plot_uk_outline : bool, default False
        If True, draw the UK grid outline first in thin black.
    """
    # Defensive programming
    if not isinstance(first_dataframe, pd.DataFrame):
        raise TypeError(f"first_dataframe must be a DataFrame, got {type(first_dataframe)}")
    if not isinstance(second_dataframe, pd.DataFrame):
        raise TypeError(f"second_dataframe must be a DataFrame, got {type(second_dataframe)}")
    if air_pollutant not in first_dataframe.columns:
        raise ValueError(f"{air_pollutant!r} not in first_dataframe")
    if air_pollutant not in second_dataframe.columns:
        raise ValueError(f"{air_pollutant!r} not in second_dataframe")

    # Compute change
    change_gdf = first_dataframe[["geometry"]].copy(deep=True)
    change_gdf[f"{air_pollutant} Change"] = (
        first_dataframe[air_pollutant] - second_dataframe[air_pollutant]
    )

    overall_min = change_gdf[f"{air_pollutant} Change"].min()
    overall_max = change_gdf[f"{air_pollutant} Change"].max()
    divnorm = colors.TwoSlopeNorm(vmin=overall_min, vcenter=0, vmax=overall_max)

    # Plot
    fig, ax = _create_base_axes(plot_uk_outline, figsize=(5, 5))
    change_gdf.plot(
        ax=ax,
        column=f"{air_pollutant} Change",
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
    _save_figure(fig, filename)


def change_in_aqi_visulisation(
    first_dataframe, second_dataframe, air_pollutant, filename, plot_uk_outline: bool = False,
):
    """
    Visualisation the change in concentrations for two datasets of air pollution
    concentrations based on air quality indexes.

    Parameters:
    -----------
    first_dataframe : pandas.DataFrame
    second_dataframe : pandas.DataFrame
    air_pollutant : str
    filename : str
    plot_uk_outline : bool, default False
    """
    # Defensive programming
    if not isinstance(first_dataframe, pd.DataFrame):
        raise TypeError(f"first_dataframe must be a DataFrame, got {type(first_dataframe)}")
    if not isinstance(second_dataframe, pd.DataFrame):
        raise TypeError(f"second_dataframe must be a DataFrame, got {type(second_dataframe)}")
    if air_pollutant not in first_dataframe.columns:
        raise ValueError(f"{air_pollutant!r} not in first_dataframe")
    if air_pollutant not in second_dataframe.columns:
        raise ValueError(f"{air_pollutant!r} not in second_dataframe")

    # Prepare change
    change_gdf = first_dataframe[["geometry"]].copy(deep=True)
    first = first_dataframe.copy()
    second = second_dataframe.copy()
    first[air_pollutant] = first[air_pollutant].astype(int)
    second[air_pollutant] = second[air_pollutant].astype(int)
    change_gdf[f"{air_pollutant} Change"] = first[air_pollutant] - second[air_pollutant]

    overall_min = change_gdf[f"{air_pollutant} Change"].min()
    overall_max = change_gdf[f"{air_pollutant} Change"].max()
    overall_min = min(overall_min, -1)
    overall_max = max(overall_max, 1)
    divnorm = colors.TwoSlopeNorm(vmin=overall_min, vcenter=0, vmax=overall_max)

    # Plot
    fig, ax = _create_base_axes(plot_uk_outline, figsize=(5, 5))
    change_gdf.plot(
        ax=ax,
        column=f"{air_pollutant} Change",
        vmin=overall_min,
        vmax=overall_max,
        cmap="bwr",
        norm=divnorm,
        legend=True,
        legend_kwds={"format": "%.0f", "shrink": 0.5, "label": "AQI Change"},
    )
    _save_figure(fig, filename)


def change_in_concentration_line(
    air_pollutant, baseline_list, change_list, days, hours_covered, filename
):
    """
    Visualisation the change in concentrations for two datasets of air pollution concentrations
    in a line graph.

    Parameters:
    -----------
    air_pollutant : str
        The name of the air pollutant to plot,
    baseline_list : list
        List of the air pollution concentrations for the baseline scenario.
    change_list : list
        List of the air pollution concentrations for the future scenario.
    days : list
        The days the lists cover.
    hours_covered : list
        The hours the lists cover.
    filename : str
        Filename for the visualisation output.
    """
    lineStyle = {"Mean": "solid", "Max": "dotted", "Total": "dashed"}
    fig, axes = plt.subplots(1, figsize=(15, 5))

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
            timestamp_labels.append(f"{day_of_week} {hour}")
            hours.append(hour_counter)
            hour_counter += 1

            baseline_pollution_data.append(
                baseline_pollution_hour["Model Prediction"].mean()
            )
            updated_pollution_data.append(
                change_pollution_hour["Model Prediction"].mean()
            )

    single_pollutant_data = pd.DataFrame.from_dict({
        "Labels": timestamp_labels,
        "Hour": hours,
        "Baseline Pollution": baseline_pollution_data,
        "Update Pollution": updated_pollution_data,
    })

    # Smooth the line by interpolation
    for _ in range(10):
        N = 1
        single_pollutant_data.index = single_pollutant_data.index * (N + 1)
        single_pollutant_data = single_pollutant_data.reindex(
            np.arange(single_pollutant_data.index.max() + N + 1)
        )[:-1]
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


    # convert the “Hour” column to a plain numpy array first
    hour_arr = single_pollutant_data["Hour"].to_numpy()
    xx = np.repeat(hour_arr, 2)[1:]
    yy1 = np.repeat(single_pollutant_data["Baseline Pollution"].to_numpy(), 2)[:-1]
    yy2 = np.repeat(single_pollutant_data["Update Pollution"].to_numpy(), 2)[:-1]
    
    axes.set_xlim(xx[0], xx[-1])

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

    # Plot baseline line
    axes.plot(
        single_pollutant_data["Hour"],
        single_pollutant_data["Baseline Pollution"],
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
    
    orig_positions = hours               # e.g. [0,1,2,…,23, 24,25,…] if multiple days
    orig_labels    = timestamp_labels    # same length

    step = 3
    
    tick_positions = orig_positions[::step]
    tick_labels    = orig_labels[::step]
    
    axes.set_xticks(tick_positions)
    axes.set_xticklabels(
        tick_labels,
        rotation=60,
        ha="right",
        fontsize=10,
    )
    axes.set_xlim(xx[0], xx[-1])
    
    fig.tight_layout()




    axes.legend(loc="upper left")
    fig.savefig(VISUALISATIONS_DIR / f"{filename}.png", bbox_inches="tight")
    if matplotlib.is_interactive():
        plt.show()
    plt.close(fig)
