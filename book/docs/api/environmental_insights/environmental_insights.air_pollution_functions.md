<a id="environmental_insights.air_pollution_functions"></a>

# environmental\_insights.air\_pollution\_functions

<a id="environmental_insights.air_pollution_functions.air_pollution_concentrations_to_UK_daily_air_quality_index"></a>

#### air\_pollution\_concentrations\_to\_UK\_daily\_air\_quality\_index

```python
def air_pollution_concentrations_to_UK_daily_air_quality_index(
        predicitions, pollutant, air_pollutant_column_name)
```

Add onto an existing dataframe the Daily Air Quality Index
(https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
for the air pollutant concentration data described.

**Arguments**:

  -----------
  predicitions : pandas.DataFrame
  A dataframe of the air pollution concentrations that are to be added onto.
  pollutant : str
  The string of the air pollutant concentration thresholds to be used
  to create the air quality indexes.
  air_pollutant_column_name : str
  The string of the column name for the air pollution concentration
  to calculate the air quality index on.
  

**Returns**:

  --------
  pandas.DataFrame
  A dataframe with the additional columns for the air quality index
  based on the outlined air pollution concentration data.

<a id="environmental_insights.air_pollution_functions.visualise_air_pollution_daily_air_quality_index"></a>

#### visualise\_air\_pollution\_daily\_air\_quality\_index

```python
def visualise_air_pollution_daily_air_quality_index(
        air_pollution_GDF,
        aqi_to_plot,
        filename,
        plot_uk_outline: bool = False)
```

Visualise air_pollution_GDF with the UK Daily Air Quality Index
(https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
using the individual index bounds and standard color codes.

**Arguments**:

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

<a id="environmental_insights.air_pollution_functions.visualise_air_pollution_daily_air_quality_bands"></a>

#### visualise\_air\_pollution\_daily\_air\_quality\_bands

```python
def visualise_air_pollution_daily_air_quality_bands(
        air_pollution_GDF,
        aqi_to_plot,
        filename,
        plot_uk_outline: bool = False)
```

Visualise air_pollution_GDF with the UK Daily Air Quality Index bands
(https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info)
using the band names and standard color codes.

**Arguments**:

  -----------
  air_pollution_GDF : geopandas.GeoDataFrame
  aqi_to_plot : str
  Name of the column with AQI bands. Must be one of:
  ["no2 Air Quality Index AQI Band", "o3 Air Quality Index AQI Band",
  "pm10 Air Quality Index AQI Band", "pm2p5 Air Quality Index AQI Band",
  "so2 Air Quality Index AQI Band"].
  filename : str
  plot_uk_outline : bool, default False

<a id="environmental_insights.air_pollution_functions.change_in_concentrations_visulisation"></a>

#### change\_in\_concentrations\_visulisation

```python
def change_in_concentrations_visulisation(first_dataframe,
                                          second_dataframe,
                                          air_pollutant,
                                          filename,
                                          plot_uk_outline: bool = False)
```

Visualisation the change in concentrations for two datasets of air pollution
concentrations based on actual concentrations.

**Arguments**:

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

<a id="environmental_insights.air_pollution_functions.change_in_aqi_visulisation"></a>

#### change\_in\_aqi\_visulisation

```python
def change_in_aqi_visulisation(first_dataframe,
                               second_dataframe,
                               air_pollutant,
                               filename,
                               plot_uk_outline: bool = False)
```

Visualisation the change in concentrations for two datasets of air pollution
concentrations based on air quality indexes.

**Arguments**:

  -----------
  first_dataframe : pandas.DataFrame
  second_dataframe : pandas.DataFrame
  air_pollutant : str
  filename : str
  plot_uk_outline : bool, default False

<a id="environmental_insights.air_pollution_functions.change_in_concentration_line"></a>

#### change\_in\_concentration\_line

```python
def change_in_concentration_line(air_pollutant, baseline_list, change_list,
                                 days, hours_covered, filename)
```

Visualisation the change in concentrations for two datasets of air pollution concentrations
in a line graph.

**Arguments**:

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

