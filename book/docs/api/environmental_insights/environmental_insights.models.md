<a id="environmental_insights.models"></a>

# environmental\_insights.models

<a id="environmental_insights.models.ensure_download"></a>

#### ensure\_download

```python
def ensure_download(dataset: str,
                    data_type: str,
                    *,
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

Ensure the requested file(s) exist locally by invoking the unified downloader.

<a id="environmental_insights.models.load_model_united_kingdom"></a>

#### load\_model\_united\_kingdom

```python
def load_model_united_kingdom(
        model_level: str,
        pollutant: str,
        model_category: str,
        token: Optional[str] = None) -> lgb.LGBMRegressor
```

Load a pretrained UK air pollution LGBM model for a specific category,
downloading only the required booster + params.

Model directory structure (identical for both namespaces):
  {namespace}/Models/{level}/{pollutant}/All_Stations/{pollutant}_{category}
but when model_category == "All", use just {pollutant} with no suffix.

Namespaces:
  - ML-HAPPE          (for existing categories)
  - SynthHAPPE_v2     (for Climate_Projections_Models and Transport_Infrastructure_Policy_Models)

<a id="environmental_insights.models.load_model_global"></a>

#### load\_model\_global

```python
def load_model_global(model_level: str,
                      pollutant: str,
                      model_category: str,
                      token: Optional[str] = None) -> lgb.LGBMRegressor
```

Load a pretrained GLOBAL ML-HAPPG air pollution LGBM model for a specific category,
downloading only the required booster + params.

For 'temporal' category:
    Models/{model_level}/temporal/{pollutant}
For all other categories:
    Models/{model_level}/{pollutant}/All_Stations/{suffix}
    where suffix = pollutant if model_category == 'All'
                  else f"{pollutant}_{model_category}"

<a id="environmental_insights.models.make_concentration_predictions_united_kingdom"></a>

#### make\_concentration\_predictions\_united\_kingdom

```python
def make_concentration_predictions_united_kingdom(
        estimating_model: lgb.LGBMRegressor, observation_data: pd.DataFrame,
        feature_names: List[str]) -> pd.DataFrame
```

Predict concentrations using a trained LGBM model.

<a id="environmental_insights.models.make_concentration_predictions_global"></a>

#### make\_concentration\_predictions\_global

```python
def make_concentration_predictions_global(
        estimating_model: lgb.LGBMRegressor, observation_data: pd.DataFrame,
        feature_names: List[str]) -> pd.DataFrame
```

Predict concentrations using a trained GLOBAL LGBM model.

<a id="environmental_insights.models.rename_global_input_columns"></a>

#### rename\_global\_input\_columns

```python
def rename_global_input_columns(df: pd.DataFrame) -> pd.DataFrame
```

Rename columns in a DataFrame to match the GLOBAL model's expected names.

