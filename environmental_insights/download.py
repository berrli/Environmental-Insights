import os
import requests
import pandas as pd
from rpy2 import robjects


def import_monitoring_station_data(site, pollutant, years, source):
    """
    Import monitoring station data from specified source and years.

    Parameters:
    site (str): The monitoring site identifier.
    pollutant (str): The pollutant identifier.
    years (int or list of int): The year or list of years to import data for.
    source (str): The data source identifier (e.g., 'aurn', 'saqn', 'aqe', 'waqn', 'ni').

    Returns:
    DataFrame: A pandas DataFrame containing the combined data for all specified years,
               or None if any downloads failed.
    """

    # Convert site and pollutant identifiers to uppercase
    site = site.upper()
    pollutant = pollutant.upper()

    # Ensure years is a list
    if isinstance(years, int):
        years = [years]

    # Initialize variables to store downloaded data and track errors
    downloaded_data = []
    errors_raised = False

    # Dictionary mapping source identifiers to their base URLs
    source_dict = {
        "aurn": "https://uk-air.defra.gov.uk/openair/R_data/",
        "saqn": "https://www.scottishairquality.scot/openair/R_data/",
        "aqe": "https://airqualityengland.co.uk/assets/openair/R_data/",
        "waqn": "https://airquality.gov.wales/sites/default/files/openair/R_data/",
        "ni": "https://www.airqualityni.co.uk/openair/R_data/",
    }

    # Get the base URL for the specified source
    source_url = source_dict.get(source)

    for year in years:
        # Construct the URL for the RData file
        url = f"{source_url}{site}_{year}.RData"
        print(url)

        # Define request headers to simulate a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Download the RData file
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            # Save the downloaded RData file locally
            local_filename = f"temp_downloads/{site}_{year}.RData"
            with open(local_filename, "wb") as file:
                file.write(response.content)
            print(f"Download completed successfully for {site} in {year}.")
        else:
            # Handle download failure
            print(f"Failed to download the file. Status code: {response.status_code}")
            errors_raised = True
            continue

        # Load the RData file into R environment
        robjects.r["load"](local_filename)

        # Assuming the RData file has only one object and it is a data frame
        # Get the name of the loaded object
        r_data_frame = robjects.r[robjects.r.objects()[0]]

        # Convert the R data frame to a pandas data frame
        result = pd.DataFrame(
            {col: list(r_data_frame.rx2(col)) for col in r_data_frame.names}
        )

        # Save the pandas data frame to a CSV file
        result.to_csv(local_filename.replace(".RData", ".csv"), index=False)

        # Read the CSV file into a pandas data frame
        result = pd.read_csv(local_filename.replace(".RData", ".csv"))

        # Clear the R environment
        robjects.r("rm(list=ls())")

        # Append the data frame to the list of downloaded data
        downloaded_data.append(result)

    if errors_raised:
        print("Some files failed to download.")
        return None
    else:
        print("All files downloaded successfully.")
        # Concatenate all the downloaded data frames
        df = pd.concat(downloaded_data, ignore_index=True)
        return df


def monitoring_station_meta_data(monitoring_network="aurn"):
    """
    Import metadata for a specified monitoring network.

    Parameters:
    monitoring_network (str): The monitoring network identifier. Default is "aurn".

    Returns:
    DataFrame: A pandas DataFrame containing the metadata for the specified monitoring network,
               or None if the download fails.
    """

    # Dictionary mapping monitoring networks to their metadata URLs
    source_dict = {
        "aurn": "http://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData",
        "saqn": "https://www.scottishairquality.scot/openair/R_data/SCOT_metadata.RData",
        "aqe": "https://airqualityengland.co.uk/assets/openair/R_data/AQE_metadata.RData",
        "waqn": "https://airquality.gov.wales/sites/default/files/openair/R_data/WAQ_metadata.RData",
        "ni": "https://www.airqualityni.co.uk/openair/R_data/NI_metadata.RData",
    }

    # Get the URL for the specified monitoring network
    url = source_dict.get(monitoring_network)
    display(url)

    # Define request headers to simulate a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send a GET request to the URL with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Create the temp_downloads directory if it doesn't exist
        os.makedirs("temp_downloads", exist_ok=True)

        # Save the content to a local file
        local_filename = f"temp_downloads/{monitoring_network}_metadata.RData"
        with open(local_filename, "wb") as file:
            file.write(response.content)
        print("Download completed successfully.")
    else:
        # Handle download failure
        print("Failed to download the file. Status code:", response.status_code)
        return None

    # Load the RData file into R environment
    robjects.r["load"](local_filename)

    # Assuming the RData file has only one object and it is a data frame
    # Get the name of the loaded object
    r_data_frame = robjects.r[robjects.r.objects()[0]]

    # Convert the R data frame to a pandas data frame
    result = pd.DataFrame(
        {col: list(r_data_frame.rx2(col)) for col in r_data_frame.names}
    )

    # Save the pandas data frame to a CSV file
    result.to_csv(local_filename.replace(".RData", ".csv"), index=False)

    # Clear the R environment
    robjects.r("rm(list=ls())")

    # Read the CSV file into a pandas data frame
    meta_data = pd.read_csv(local_filename.replace(".RData", ".csv"))
    return meta_data
