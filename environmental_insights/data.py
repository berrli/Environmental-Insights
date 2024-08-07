import geopandas as gpd
import pandas as pd
import os
import overpy
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
import numpy as np
import itertools
from operator import itemgetter

pd.options.mode.chained_assignment = None  # default='warn'

# Get the directory where this script is located
script_dir = os.path.dirname(__file__)


def download_file_data(filename):
    """
    Checks if a file that has been requested has been downloaded and if it has then it will download the file.

    Parameters:
    filename (string): The dataset filename to be downloaded from the remote server
    """
    print("Checking existence of file: " + str(filename))


def air_pollution_concentration_typical_day_real_time_united_kingdom(
    month, day_of_Week, hour
):
    """
    Retrieve the typical day complete dataset for the UK for a given time.

    Parameters:
    month (int): An int to represent the month of interest, 1 being January, and 12 being December.
    day_of_Week (string): A string to represent the day of the week of interest in the form of "Friday".
    hour (int): An int to represent the hour of interest, 0 being midnight, and 23 being the final possible hour of the day.

    Returns:
    DataFrame: A DataFrame of the typical dataset for the UK for a given time of interest.
    """
    # Construct the full path to the data file
    desired_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "air_pollution",
        "uk_typical_day",
        f"Month_{month}-Day_{day_of_Week}-Hour_{hour}.feather",
    )
    if not os.path.isfile(desired_filename):
        # download_file_data(desired_filename)
        pass

    air_pollution_data = pd.read_feather(desired_filename)
    air_pollution_data = air_pollution_data.rename(
        columns={"Grid ID": "UK Model Grid ID"}
    )
    return air_pollution_data


def air_pollution_concentration_nearest_point_typical_day_united_kingdom(
    month, day_of_Week, hour, latitude, longitude, uk_grids
):
    """
    Retrieve a single air pollution concentration data point predicted based on the UK data, based on the closest point given by the latitude and longitude.

    Parameters:
    month (int): An int to represent the month of interest, 1 being January, and 12 being December.
    day_of_Week (string): A string to represent the day of the week of interest in the form of "Friday".
    hour (int): An int to represent the hour of interest, 0 being midnight, and 23 being the final possible hour of the day.
    latitude (float): A float denoting the desired latitude.
    longitude (float): A float denoting the desired longitude.
    uk_grids (GeoDataFrame): A GeoDataFrame that describes the estimation points for the UK model.

    Returns:
    DataFrame: A DataFrame of the nearest point in the data at the given timestamp.
    """
    # Construct the full path to the data file
    desired_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "air_pollution",
        "uk_typical_day",
        f"Month_{month}-Day_{day_of_Week}-Hour_{hour}.feather",
    )
    if not os.path.isfile(desired_filename):
        # download_file_data(desired_filename)
        pass

    air_pollution_data = pd.read_feather(desired_filename)
    air_pollution_data = air_pollution_data.rename(
        columns={"Grid ID": "UK Model Grid ID"}
    )
    air_pollution_data = uk_grids.merge(air_pollution_data, on="UK Model Grid ID")
    air_pollution_data["geometry"] = air_pollution_data["geometry"].centroid
    air_pollution_data = air_pollution_data.to_crs(4326)
    air_pollution_data["Latitude"] = air_pollution_data["geometry"].y
    air_pollution_data["Longitude"] = air_pollution_data["geometry"].x

    tree = cKDTree(air_pollution_data[["Latitude", "Longitude"]])

    # Query for the closest points
    distance, idx = tree.query([latitude, longitude], k=1)

    # Retrieve the closest points
    closest_points = pd.DataFrame(air_pollution_data.iloc[idx]).T
    closest_points["Distance"] = distance
    closest_points = closest_points.rename(
        columns={"Latitude": "Prediction Latitude", "Longitude": "Prediction Longitude"}
    )
    closest_points["Requested Latitude"] = latitude
    closest_points["Requested Longitude"] = longitude
    closest_points = closest_points.drop(columns=["UK Model Grid ID", "geometry"])
    return closest_points


def air_pollution_concentration_complete_set_real_time_united_kingdom(time):
    """
    Retrieve the complete predicted dataset for a given timestamp in the UK dataset.

    Parameters:
    time (string): A string denoting the timestamp desired, of the form YYYY-MM-DD HHmmss.

    Returns:
    DataFrame: A DataFrame of the dataset for the UK for a given timestamp.
    """
    # Construct the full path to the data file
    desired_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "air_pollution",
        "uk_complete_set",
        f"{time}.feather",
    )
    if not os.path.isfile(desired_filename):
        download_file_data(time)

    air_pollution_data = pd.read_feather(desired_filename)
    air_pollution_data = air_pollution_data.rename(
        columns={"Grid ID": "UK Model Grid ID"}
    )
    return air_pollution_data


def air_pollution_concentration_nearest_point_real_time_united_kingdom(
    latitude, longitude, time, uk_grids
):
    """
    Retrieve a single air pollution concentration data point predicted based on the UK data, based on the closest point given by the latitude and longitude.

    Parameters:
    latitude (float): A float denoting the desired latitude.
    longitude (float): A float denoting the desired longitude.
    time (string): A string denoting the timestamp desired, of the form YYYY-MM-DD HHmmss.
    uk_grids (GeoDataFrame): A GeoDataFrame that describes the estimation points for the UK model.

    Returns:
    DataFrame: A DataFrame of the nearest point in the data at the given timestamp.
    """
    print(
        "Accessing air pollution concentration at: Latitude: "
        + str(latitude)
        + " Longitude: "
        + str(longitude)
        + " Time: "
        + str(time)
    )

    # Construct the full path to the data file
    desired_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "air_pollution",
        "uk_complete_set",
        f"{time}.feather",
    )
    if not os.path.isfile(desired_filename):
        download_file_data(time)

    air_pollution_data = pd.read_feather(desired_filename)
    air_pollution_data = air_pollution_data.rename(
        columns={"Grid ID": "UK Model Grid ID"}
    )
    air_pollution_data = uk_grids.merge(air_pollution_data, on="UK Model Grid ID")
    air_pollution_data["geometry"] = air_pollution_data["geometry"].centroid

    air_pollution_data = air_pollution_data.to_crs(4326)
    air_pollution_data["Latitude"] = air_pollution_data["geometry"].y
    air_pollution_data["Longitude"] = air_pollution_data["geometry"].x

    tree = cKDTree(air_pollution_data[["Latitude", "Longitude"]])

    # Query for the closest points
    distance, idx = tree.query([latitude, longitude], k=1)

    # Retrieve the closest points
    closest_points = pd.DataFrame(air_pollution_data.iloc[idx]).T
    closest_points["Distance"] = distance
    closest_points = closest_points.rename(
        columns={"Latitude": "Prediction Latitude", "Longitude": "Prediction Longitude"}
    )
    closest_points["Requested Latitude"] = latitude
    closest_points["Requested Longitude"] = longitude
    closest_points = closest_points.drop(columns=["UK Model Grid ID", "geometry"])
    return closest_points


def air_pollution_concentration_complete_set_real_time_global(time):
    """
    Retrieve the complete calculated dataset for a given timestamp in the global dataset.

    Parameters:
    time (string): A string denoting the timestamp desired, of the form DD-MM-YYYY HHmmss.

    Returns:
    DataFrame: A DataFrame of the dataset for the global model for a given timestamp.
    """
    # Construct the full path to the data file
    desired_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "air_pollution",
        "global_complete_set",
        f"{time}.feather",
    )
    if not os.path.isfile(desired_filename):
        download_file_data(time)

    air_pollution_data = pd.read_feather(desired_filename)
    air_pollution_data = air_pollution_data.rename(
        columns={"id": "Global Model Grid ID"}
    )
    return air_pollution_data


def get_amenities_as_geodataframe(amenity_type, min_lat, min_lon, max_lat, max_lon):
    """
    Fetch amenities of a given type within a bounding box and return as a GeoDataFrame.

    Parameters:
    amenity_type (string): Type of amenity, e.g., "hospital".
    min_lat (float): Minimum latitude.
    min_lon (float): Minimum longitude.
    max_lat (float): Maximum latitude.
    max_lon (float): Maximum longitude.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the amenities with their names and coordinates.
    """
    api = overpy.Overpass()

    # Define the Overpass query
    query = f"""
    [out:json];
    (
        node["amenity"="{amenity_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["amenity"="{amenity_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
        relation["amenity"="{amenity_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """

    result = api.query(query)

    # Extract results and store them in lists
    names = []
    lats = []
    lons = []

    for node in result.nodes:
        names.append(node.tags.get("name", "Unknown"))
        lats.append(node.lat)
        lons.append(node.lon)

    for way in result.ways:
        names.append(way.tags.get("name", "Unknown"))
        lats.append(way.center_lat)
        lons.append(way.center_lon)

    for relation in result.relations:
        names.append(relation.tags.get("name", "Unknown"))
        lats.append(relation.center_lat)
        lons.append(relation.center_lon)

    # Convert lists to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(lons, lats)]
    gdf = gpd.GeoDataFrame({"name": names, "geometry": geometry})

    return gdf


def get_highways_as_geodataframe(highway_type, min_lat, min_lon, max_lat, max_lon):
    """
    Fetch highways of a specified type within a bounding box from OSM and return as a GeoDataFrame.

    Parameters:
    highway_type (string): Type of highway, e.g., "motorway", "residential".
    min_lat (float): Minimum latitude.
    min_lon (float): Minimum longitude.
    max_lat (float): Maximum latitude.
    max_lon (float): Maximum longitude.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the highways with their names and coordinates.
    """
    api = overpy.Overpass()

    # Modify the Overpass query to retrieve nodes of ways explicitly
    query = f"""
    [out:json];
    (
        way["highway"="{highway_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
        >;  // Fetches all nodes for the ways returned in the previous statement
    );
    out geom;
    """

    result = api.query(query)

    names = []
    geometries = []

    for way in result.ways:
        try:
            coords = [(node.lon, node.lat) for node in way.nodes]
            names.append(way.tags.get("name", "Unknown"))
            geometries.append(LineString(coords))
        except Exception as e:
            print(f"Error processing way: {way.id}. Error: {e}")

    # Convert lists to a GeoDataFrame
    gdf = gpd.GeoDataFrame({"name": names, "geometry": geometries}, crs=4326)
    gdf["highway"] = highway_type
    gdf["source"] = "osm"
    return gdf


def ckd_nearest_LineString(gdf_A, gdf_B, gdf_B_cols):
    """
    Calculate the nearest points between two GeoDataFrames containing LineString geometries.

    This function uses cKDTree to efficiently find the nearest points between two sets
    of LineStrings. For each point in `gdf_A`, the function finds the closest point
    in `gdf_B` and returns the distances along with selected columns from `gdf_B`.

    Parameters:
    gdf_A (GeoDataFrame): A GeoDataFrame containing LineString geometries.
    gdf_B (GeoDataFrame): A GeoDataFrame containing LineString geometries which will be
                         used to find the closest points to `gdf_A`.
    gdf_B_cols (list or tuple): A list or tuple containing column names from `gdf_B`
                               which will be included in the resulting DataFrame.

    Returns:
    GeoDataFrame: A GeoDataFrame with each row containing a geometry from `gdf_A`,
                  corresponding closest geometry details from `gdf_B` (as specified by `gdf_B_cols`),
                  and the distance to the closest point in `gdf_B`.

    Note:
    The resulting GeoDataFrame maintains the order of `gdf_A` and attaches the nearest
    details from `gdf_B`.
    This code was adapted from the code available here: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    """
    gdf_A = gdf_A.reset_index(drop=True)
    gdf_B = gdf_B.reset_index(drop=True)
    A = np.concatenate([np.array(geom.coords) for geom in gdf_A.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdf_B.geometry.to_list()]
    B_ix = tuple(
        itertools.chain.from_iterable(
            [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]
        )
    )
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [
            gdf_A,
            gdf_B.loc[idx, gdf_B_cols].reset_index(drop=True),
            pd.Series(dist, name="dist"),
        ],
        axis=1,
    )
    return gdf


def get_even_spaced_points(points, number_of_points):
    """
    Generate a list of evenly spaced points between two given points.

    This function calculates the distance (or difference) between two input points
    and divides this distance into `number_of_points` equal segments. The resulting
    points, including the start and end points, are returned in a list.

    Parameters:
    points (list or tuple of float): A list or tuple containing two points
                                    (start and end) between which the evenly spaced points
                                    are to be calculated.
    number_of_points (int): The total number of points to generate,
                          including the start and end points.

    Returns:
    list: A list of evenly spaced points between the provided start and end points.

    Example:
    >>> get_even_spaced_points([1, 10], 5)
    [1.0, 3.25, 5.5, 7.75, 10.0]

    Note:
    The function assumes that the points list is sorted in ascending order.
    """
    step = (points[1] - points[0]) / (number_of_points - 1)

    return [points[0] + step * i for i in range(number_of_points)]


def calculate_new_metrics_distance_total(
    current_infrastructure,
    highway_type,
    start_point,
    end_point,
    land_grids_centroids,
    land_grids,
):
    """
    Simulate the addition of a proposed highway to current infrastructure and calculate new metrics.

    This function creates a new proposed highway segment based on given start and end points.
    The proposed highway is then added to the current infrastructure dataset. After adding the
    new highway, the function calculates distance metrics and total length of the specific highway type.

    Parameters:
    current_infrastructure (GeoDataFrame): The current infrastructure dataset with existing highways.
    highway_type (str): Type of the highway for which metrics are calculated (e.g., "motorway").
    start_point (tuple of float): Coordinates (x, y) for the starting point of the proposed highway.
    end_point (tuple of float): Coordinates (x, y) for the ending point of the proposed highway.
    land_grids_centroids (GeoDataFrame): GeoDataFrame of the grids for predictions to be made on, with the geometry being a set of points representing the centroid of such grids.
    land_grids (GeoDataFrame): GeoDataFrame of the grids for predictions to be made on, with the geometry being a set of polygons representing the grids themselves.

    Returns:
    tuple:
      - GeoDataFrame: Contains metrics such as road infrastructure distance and total road length for each grid.
      - GeoDataFrame: A merged dataset of current infrastructure and the proposed highway.

    Note:
    - The function assumes the use of EPSG:4326 and EPSG:3395 for coordinate reference systems.
    - It also assumes the existence of helper functions like `get_even_spaced_points` and a global variable `land_grids_centroids`.
    """
    xPoints = get_even_spaced_points([start_point[0], end_point[0]], 1000)
    yPoints = get_even_spaced_points([start_point[1], end_point[1]], 1000)

    inputCoordinates = list(map(lambda x, y: Point(x, y), xPoints, yPoints))

    proposed_highway = gpd.GeoDataFrame(
        index=[0], crs="epsg:4326", geometry=[LineString(inputCoordinates)]
    )
    proposed_highway["source"] = "User Added"
    proposed_highway["highway"] = "motorway"
    proposed_highway = proposed_highway.to_crs(3395)

    current_infrastructure_user_added = pd.concat(
        [current_infrastructure, proposed_highway]
    )

    current_infrastructure_highway_type = current_infrastructure_user_added[
        current_infrastructure_user_added["highway"] == highway_type
    ]
    current_infrastructure_highway_type = current_infrastructure_highway_type.to_crs(
        3395
    )

    # Calculate the new distance
    current_infrastructure_highway_distance = ckd_nearest_LineString(
        land_grids_centroids,
        current_infrastructure_highway_type,
        gdf_B_cols=["source", "highway"],
    )
    current_infrastructure_highway_distance = (
        current_infrastructure_highway_distance.rename(
            columns={"dist": "Road Infrastructure Distance " + str(highway_type)}
        )
    )

    # Calculate the new motorway column
    roadGrids_intersection_OSM = gpd.overlay(
        current_infrastructure_highway_type, land_grids, how="intersection"
    )

    roadGrids_intersection_OSM_Subset = roadGrids_intersection_OSM[
        ["highway", "UK Model Grid ID", "geometry"]
    ]
    roadGrids_intersection_OSM_Subset["Road Length"] = (
        roadGrids_intersection_OSM_Subset["geometry"].length
    )
    goupby_result = pd.DataFrame(
        roadGrids_intersection_OSM_Subset.groupby(["highway", "UK Model Grid ID"])[
            "Road Length"
        ].sum()
    ).reset_index()
    current_infrastructure_new_grid_total = goupby_result.pivot_table(
        values="Road Length", index="UK Model Grid ID", columns="highway", aggfunc="sum"
    )
    current_infrastructure_new_grid_total = (
        current_infrastructure_new_grid_total.rename(
            columns={highway_type: "Total Length " + str(highway_type)}
        )
    )
    current_infrastructure_all_grids = pd.merge(
        land_grids,
        current_infrastructure_new_grid_total,
        left_on="UK Model Grid ID",
        right_index=True,
        how="left",
    )
    current_infrastructure_all_grids = current_infrastructure_all_grids.fillna(0)

    return (
        current_infrastructure_all_grids.merge(
            current_infrastructure_highway_distance.drop(columns="geometry"),
            on="UK Model Grid ID",
            how="left",
        ),
        current_infrastructure_user_added,
    )


def replace_feature_vector_column(
    feature_vector, new_feature_vector, feature_vector_name
):
    """
    Replace the feature vector column name with the new feature vector column name, replacing the data within the dataframe with new environmental conditions.

    Parameters:
    feature_vector (DataFrame): DataFrame of the original data.
    new_feature_vector (DataFrame): DataFrame containing the new feature vector that is to be used to replace the data in feature_vector.
    feature_vector_name (string): Name of the feature vector to be changed.

    Returns:
    DataFrame: A DataFrame of the original data that was added with the feature vector now replaced by the new data.
    """
    feature_vector = feature_vector.drop(columns=[feature_vector_name])
    feature_vector = feature_vector.merge(
        new_feature_vector[["UK Model Grid ID", feature_vector_name]],
        on="UK Model Grid ID",
    )

    return feature_vector


def get_uk_grids():
    """
    Get the spatial grids that represent the locations at which air pollution estimations are made for the UK Model.

    Returns:
    GeoDataFrame: A GeoDataFrame of the polygons for each of the grids in the UK Model alongside their centroid and unique ID.
    """
    # Construct the full path to the data file
    grid_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "supporting_data",
        "1km_uk_grids.gpkg",
    )
    uk_grids = gpd.read_file(grid_filename)
    uk_grids["geometry Centroid"] = uk_grids["geometry"].centroid
    uk_grids = uk_grids.rename(columns={"Grid ID": "UK Model Grid ID"})
    return uk_grids


def get_global_grids():
    """
    Get the spatial grids that represent the locations at which air pollution estimations are made for the Global Model.

    Returns:
    GeoDataFrame: A GeoDataFrame of the polygons for each of the grids in the Global Model and unique ID.
    """
    # Construct the full path to the data file
    grid_filename = os.path.join(
        script_dir,
        "environmental_insights_data",
        "supporting_data",
        "025latlong_world_grids.gpkg",
    )
    global_grids = gpd.read_file(grid_filename)
    global_grids = global_grids.rename(columns={"id": "Global Model Grid ID"})
    return global_grids
