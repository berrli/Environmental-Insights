<a id="environmental_insights.download"></a>

# environmental\_insights.download

<a id="environmental_insights.download.download_file"></a>

#### download\_file

```python
def download_file(url: str,
                  output_dir: Optional[Union[str, Path]] = None,
                  token: Optional[str] = None,
                  extra_wget_args: Optional[List[str]] = None) -> None
```

Download a single URL using wget into output_dir (defaults to package root).

<a id="environmental_insights.download.download_time_point_ml"></a>

#### download\_time\_point\_ml

```python
def download_time_point_ml(
        dataset: str,
        data_type: str,
        timestamp: str,
        token: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None) -> None
```

Download a single .nc time point for ML-HAPPE Input/Output.

<a id="environmental_insights.download.download_time_point_synth"></a>

#### download\_time\_point\_synth

```python
def download_time_point_synth(
        data_type: str,
        month: Union[int, str],
        day: str,
        hour: Union[int, str],
        token: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None) -> None
```

Download a single .nc file for SynthHAPPE with synthetic filename format:
Month_<n>-Day_<DayName>-Hour_<h>.nc
Month can be numeric or month name.

<a id="environmental_insights.download.download_training_data"></a>

#### download\_training\_data

```python
def download_training_data(
        dataset: str,
        pollutant: str,
        station: str,
        token: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None) -> None
```

Download ML-HAPPE or ML-HAPPG training-data .nc for a single station.

<a id="environmental_insights.download.get_training_station_names"></a>

#### get\_training\_station\_names

```python
def get_training_station_names(dataset: str, pollutant: str) -> list[str]
```

Return a sorted list of station names (without '.nc') for
ML-HAPPE or ML-HAPPG under Training_Data/{pollutant}/.

<a id="environmental_insights.download.download"></a>

#### download

```python
def download(dataset: str,
             data_type: str,
             timestamp: Optional[str] = None,
             month: Optional[Union[int, str]] = None,
             day: Optional[str] = None,
             hour: Optional[Union[int, str]] = None,
             model_level: Optional[str] = None,
             pollutant: Optional[str] = None,
             station: Optional[str] = None,
             token: Optional[str] = None,
             output_dir: Optional[Union[str, Path]] = None) -> None
```

Unified download interface for ML-HAPPE and SynthHAPPE.

