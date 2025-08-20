<a id="environmental_insights.data"></a>

# environmental\_insights.data

<a id="environmental_insights.data.read_nc"></a>

#### read\_nc

```python
def read_nc(filepath: str) -> xr.Dataset
```

Read a NetCDF (.nc) file into an xarray.Dataset.

Parameters
----------
filepath : str
    Path to the NetCDF file.

Returns
-------
xr.Dataset

<a id="environmental_insights.data.netcdf_to_dataframe"></a>

#### netcdf\_to\_dataframe

```python
def netcdf_to_dataframe(ds: xr.Dataset) -> pd.DataFrame
```

Convert an xarray.Dataset to a pandas DataFrame,
dropping any rows that have no valid data in *any* variable.

Parameters
----------
ds : xr.Dataset
    The dataset to convert.

Returns
-------
pd.DataFrame

<a id="environmental_insights.data.get_uk_monitoring_station"></a>

#### get\_uk\_monitoring\_station

```python
def get_uk_monitoring_station(pollutant: str,
                              station: str) -> gpd.GeoDataFrame
```

Download (if needed) and load ML-HAPPE training data for a single UK monitoring station.

Parameters
----------
pollutant : str
    One of the POLLUTANTS in ML-HAPPE (e.g. "no2").
station : str
    Name of the station (without “.nc”).
token : str, optional
    Your CEDA API token, if required.

Returns
-------
GeoDataFrame
    The station’s training data as a GeoDataFrame, with a Point geometry
    constructed from its latitude/longitude.

<a id="environmental_insights.data.get_uk_monitoring_stations"></a>

#### get\_uk\_monitoring\_stations

```python
def get_uk_monitoring_stations(pollutant: str) -> List[str]
```

Ensure the local ML-HAPPE Training_Data folder exists for a pollutant,
then return the list of all station names by delegating to download.py.

<a id="environmental_insights.data.air_pollution_concentration_typical_day_real_time_united_kingdom"></a>

#### air\_pollution\_concentration\_typical\_day\_real\_time\_united\_kingdom

```python
def air_pollution_concentration_typical_day_real_time_united_kingdom(
        month: int, day_of_week: str, hour: int, data_type: str = "Input")
```

Retrieve the typical-day dataset (either Input or Output) for the UK for a given time.

**Arguments**:

  -----------
  month : int
  Month of interest, 1 (January) through 12 (December).
  day_of_week : str
  Day of week of interest, e.g., "Monday", "Tuesday", etc.
  hour : int
  Hour of interest, 0 (midnight) through 23.
  data_type : str, optional
  Whether to fetch the "Input" or the "Output" version of the SynthHAPPE dataset.
  Defaults to "Input".
  

**Returns**:

  --------
  pandas.DataFrame
  The typical-day air pollution dataset for the UK at the specified time,
  as a flattened DataFrame.

<a id="environmental_insights.data.air_pollution_concentration_nearest_point_typical_day_united_kingdom"></a>

#### air\_pollution\_concentration\_nearest\_point\_typical\_day\_united\_kingdom

```python
def air_pollution_concentration_nearest_point_typical_day_united_kingdom(
        month: int,
        day_of_week: str,
        hour: int,
        latitude: float,
        longitude: float,
        uk_grids,
        data_type: str = "Input")
```

Retrieve a single air pollution concentration data point (Input or Output)
for the UK model at the closest grid point to a given lat/long.

**Arguments**:

  -----------
  month : int
  Month of interest, 1 (January) through 12 (December).
  day_of_week : str
  Day of week of interest, e.g. "Monday", "Tuesday", etc.
  hour : int
  Hour of interest, 0 (midnight) through 23.
  latitude : float
  Latitude of the point to estimate.
  longitude : float
  Longitude of the point to estimate.
  uk_grids : GeoDataFrame
  Model grid points for the UK.
  data_type : str, optional
  "Input" or "Output" dataset to fetch. Defaults to "Input".
  

**Returns**:

  --------
  pandas.DataFrame
  One-row DataFrame with the nearest grid’s pollutant values
  at the specified time.

<a id="environmental_insights.data.air_pollution_concentration_complete_set_real_time_united_kingdom"></a>

#### air\_pollution\_concentration\_complete\_set\_real\_time\_united\_kingdom

```python
def air_pollution_concentration_complete_set_real_time_united_kingdom(
        time: str, data_type: str = "Input")
```

Retrieve the complete predicted dataset (Input or Output) for a given timestamp in the UK ML-HAPPE dataset.

**Arguments**:

  -----------
  time : str
  Timestamp of the form "YYYY-MM-DD HHmmss".
  data_type : str, optional
  Whether to fetch the "Input" or "Output" version of the ML-HAPPE dataset.
  Defaults to "Input".
  

**Returns**:

  --------
  pandas.DataFrame
  The full air pollution dataset for the UK at the specified timestamp,
  as a flattened DataFrame.

<a id="environmental_insights.data.air_pollution_concentration_nearest_point_real_time_united_kingdom"></a>

#### air\_pollution\_concentration\_nearest\_point\_real\_time\_united\_kingdom

```python
def air_pollution_concentration_nearest_point_real_time_united_kingdom(
        latitude: float,
        longitude: float,
        time: str,
        uk_grids,
        data_type: str = "Input")
```

Retrieve a single air pollution concentration data point (Input or Output)
for the UK ML-HAPPE model at the closest grid point to a given lat/long and timestamp.

**Arguments**:

  -----------
  latitude : float
  Latitude of the point to estimate.
  longitude : float
  Longitude of the point to estimate.
  time : str
  Timestamp of the form "YYYY-MM-DD HHmmss".
  uk_grids : GeoDataFrame
  Model grid points for the UK.
  data_type : str, optional
  "Input" or "Output" dataset to fetch. Defaults to "Input".
  

**Returns**:

  --------
  pandas.DataFrame
  One‐row DataFrame with the nearest grid’s pollutant values
  at the specified time.

<a id="environmental_insights.data.air_pollution_concentration_complete_set_real_time_global"></a>

#### air\_pollution\_concentration\_complete\_set\_real\_time\_global

```python
def air_pollution_concentration_complete_set_real_time_global(
        time: str, data_type: str = "Input")
```

Retrieve the complete predicted dataset (Input or Output) for a given timestamp
in the Global ML-HAPPG dataset. This mirrors the UK real-time helper but
points at ML-HAPPG and uses Global file locations.

Parameters
----------
time : str
Timestamp of the form "YYYY-MM-DD HHmmss" (e.g., "2022-01-03 120000").
Note: ensure the date exists in the ML-HAPPG archive (e.g., 2022).
data_type : {"Input","Output"}, default "Input"
Which dataset to fetch.

Returns
-------
pandas.DataFrame
The full global grid at the specified timestamp as a flattened DataFrame.

<a id="environmental_insights.data.air_pollution_concentration_nearest_point_real_time_global"></a>

#### air\_pollution\_concentration\_nearest\_point\_real\_time\_global

```python
def air_pollution_concentration_nearest_point_real_time_global(
        latitude: float,
        longitude: float,
        time: str,
        global_grids,
        data_type: str = "Input")
```

Retrieve a single air pollution concentration (Input or Output)
for the Global ML-HAPPG model at the closest grid to a given lat/long and timestamp.

Returns
-------
pandas.DataFrame
    One-row DataFrame with the nearest grid’s values.

<a id="environmental_insights.data.get_global_monitoring_station"></a>

#### get\_global\_monitoring\_station

```python
def get_global_monitoring_station(pollutant: str,
                                  station: str) -> gpd.GeoDataFrame
```

Download (if needed) and load ML-HAPPG training data for a single global monitoring station.

<a id="environmental_insights.data.get_global_monitoring_stations"></a>

#### get\_global\_monitoring\_stations

```python
def get_global_monitoring_stations(pollutant: str) -> List[str]
```

Return the list of global station names for a pollutant (ML-HAPPG).

<a id="environmental_insights.data.get_amenities_as_geodataframe"></a>

#### get\_amenities\_as\_geodataframe

```python
def get_amenities_as_geodataframe(amenity_type, min_lat, min_lon, max_lat,
                                  max_lon)
```

Fetch amenities of a given type within a bounding box and return as a GeoDataFrame.

**Arguments**:

- `amenity_type` _string_ - Type of amenity, e.g., "hospital".
- `min_lat` _float_ - Minimum latitude.
- `min_lon` _float_ - Minimum longitude.
- `max_lat` _float_ - Maximum latitude.
- `max_lon` _float_ - Maximum longitude.
  

**Returns**:

- `GeoDataFrame` - A GeoDataFrame containing the amenities with their names and coordinates.

<a id="environmental_insights.data.get_highways_as_geodataframe"></a>

#### get\_highways\_as\_geodataframe

```python
def get_highways_as_geodataframe(highway_type, min_lat, min_lon, max_lat,
                                 max_lon)
```

Fetch highways of a specified type within a bounding box from OSM and return as a GeoDataFrame.

**Arguments**:

- `highway_type` _string_ - Type of highway, e.g., "motorway", "residential".
- `min_lat` _float_ - Minimum latitude.
- `min_lon` _float_ - Minimum longitude.
- `max_lat` _float_ - Maximum latitude.
- `max_lon` _float_ - Maximum longitude.
  

**Returns**:

- `GeoDataFrame` - A GeoDataFrame containing the highways with their names and coordinates.

<a id="environmental_insights.data.ckd_nearest_LineString"></a>

#### ckd\_nearest\_LineString

```python
def ckd_nearest_LineString(gdf_A, gdf_B, gdf_B_cols)
```

Calculate the nearest points between two GeoDataFrames containing LineString geometries.

This function uses cKDTree to efficiently find the nearest points between two sets
of LineStrings. For each point in `gdf_A`, the function finds the closest point
in `gdf_B` and returns the distances along with selected columns from `gdf_B`.

**Arguments**:

- `gdf_A` _GeoDataFrame_ - A GeoDataFrame containing LineString geometries.
- `gdf_B` _GeoDataFrame_ - A GeoDataFrame containing LineString geometries which will be
  used to find the closest points to `gdf_A`.
- `gdf_B_cols` _list or tuple_ - A list or tuple containing column names from `gdf_B`
  which will be included in the resulting DataFrame.
  

**Returns**:

- `GeoDataFrame` - A GeoDataFrame with each row containing a geometry from `gdf_A`,
  corresponding closest geometry details from `gdf_B` (as specified by `gdf_B_cols`),
  and the distance to the closest point in `gdf_B`.
  

**Notes**:

  The resulting GeoDataFrame maintains the order of `gdf_A` and attaches the nearest
  details from `gdf_B`.
  This code was adapted from the code available here: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas

<a id="environmental_insights.data.get_even_spaced_points"></a>

#### get\_even\_spaced\_points

```python
def get_even_spaced_points(points, number_of_points)
```

Generate a list of evenly spaced points between two given points.

This function calculates the distance (or difference) between two input points
and divides this distance into `number_of_points` equal segments. The resulting
points, including the start and end points, are returned in a list.

**Arguments**:

- `points` _list or tuple of float_ - A list or tuple containing two points
  (start and end) between which the evenly spaced points
  are to be calculated.
- `number_of_points` _int_ - The total number of points to generate,
  including the start and end points.
  

**Returns**:

- `list` - A list of evenly spaced points between the provided start and end points.
  

**Example**:

  >>> get_even_spaced_points([1, 10], 5)
  [1.0, 3.25, 5.5, 7.75, 10.0]
  

**Notes**:

  The function assumes that the points list is sorted in ascending order.

<a id="environmental_insights.data.calculate_new_metrics_distance_total"></a>

#### calculate\_new\_metrics\_distance\_total

```python
def calculate_new_metrics_distance_total(current_infrastructure, highway_type,
                                         start_point, end_point,
                                         land_grids_centroids, land_grids)
```

Simulate the addition of a proposed highway to current infrastructure and calculate new metrics.

This function creates a new proposed highway segment based on given start and end points.
The proposed highway is then added to the current infrastructure dataset. After adding the
new highway, the function calculates distance metrics and total length of the specific highway type.

**Arguments**:

- `current_infrastructure` _GeoDataFrame_ - The current infrastructure dataset with existing highways.
- `highway_type` _str_ - Type of the highway for which metrics are calculated (e.g., "motorway").
- `start_point` _tuple of float_ - Coordinates (x, y) for the starting point of the proposed highway.
- `end_point` _tuple of float_ - Coordinates (x, y) for the ending point of the proposed highway.
- `land_grids_centroids` _GeoDataFrame_ - GeoDataFrame of the grids for predictions to be made on, with the geometry being a set of points representing the centroid of such grids.
- `land_grids` _GeoDataFrame_ - GeoDataFrame of the grids for predictions to be made on, with the geometry being a set of polygons representing the grids themselves.
  

**Returns**:

  tuple:
  - GeoDataFrame: Contains metrics such as road infrastructure distance and total road length for each grid.
  - GeoDataFrame: A merged dataset of current infrastructure and the proposed highway.
  

**Notes**:

  - The function assumes the use of EPSG:4326 and EPSG:3395 for coordinate reference systems.
  - It also assumes the existence of helper functions like `get_even_spaced_points` and a global variable `land_grids_centroids`.

<a id="environmental_insights.data.replace_feature_vector_column"></a>

#### replace\_feature\_vector\_column

```python
def replace_feature_vector_column(feature_vector, new_feature_vector,
                                  feature_vector_name)
```

Replace the feature vector column name with the new feature vector column name, replacing the data within the dataframe with new environmental conditions.

**Arguments**:

- `feature_vector` _DataFrame_ - DataFrame of the original data.
- `new_feature_vector` _DataFrame_ - DataFrame containing the new feature vector that is to be used to replace the data in feature_vector.
- `feature_vector_name` _string_ - Name of the feature vector to be changed.
  

**Returns**:

- `DataFrame` - A DataFrame of the original data that was added with the feature vector now replaced by the new data.

<a id="environmental_insights.data.get_uk_grids"></a>

#### get\_uk\_grids

```python
def get_uk_grids()
```

Get the spatial grids that represent the locations at which air pollution estimations are made for the UK Model.

**Returns**:

- `GeoDataFrame` - A GeoDataFrame of the polygons for each of the grids in the UK Model alongside their centroid and unique ID.

<a id="environmental_insights.data.get_uk_grids_outline"></a>

#### get\_uk\_grids\_outline

```python
def get_uk_grids_outline()
```

Get the spatial grid outlines representing the boundaries of the UK Model's 1km grids.

**Returns**:

- `GeoDataFrame` - A GeoDataFrame of the grid outlines for each UK Model grid alongside their centroid and unique ID.

<a id="environmental_insights.data.get_global_grids"></a>

#### get\_global\_grids

```python
def get_global_grids()
```

Get the spatial grids that represent the locations at which air pollution estimations are made for the Global Model.

**Returns**:

- `GeoDataFrame` - A GeoDataFrame of the polygons for each of the grids in the Global Model and unique ID.

