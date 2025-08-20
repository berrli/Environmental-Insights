# Environmental Insights

[![PyPI version](https://img.shields.io/pypi/v/environmental-insights.svg?cacheSeconds=3600)](https://pypi.org/project/environmental-insights)
[![GitHub release](https://img.shields.io/github/v/release/liamjberrisford/Environmental-Insights.svg?sort=semver&cacheSeconds=3600)](https://github.com/liamjberrisford/Environmental-Insights/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/liamjberrisford/Environmental-Insights/release-and-publish.yml?branch=main&cacheSeconds=3600)](https://github.com/liamjberrisford/Environmental-Insights/actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/liamjberrisford/Environmental-Insights/testing.yml?branch=main&label=tests&style=flat-square)](https://github.com/liamjberrisford/Environmental-Insights/actions/workflows/testing.yml)



A Python package for democratizing access to ambient air pollution data and predictive analytics.

---

## 📖 Description

**Environmental Insights** provides easy-to-use functions to download, process, and analyze ambient air pollution and meteorological data over England.  
- Implements supervised machine-learning pipelines to predict hourly pollutant concentrations on a 1 km² grid.  
- Supplies both “typical day” aggregates (percentiles) and full hourly model outputs.  
- Includes geospatial utilities for mapping, interpolation, and uncertainty analysis.

---

## ⚙️ Installation

Install from PyPI:

```bash
pip install environmental-insights
```

Or from source:

```bash
git clone https://github.com/liamjberrisford/Environmental-Insights.git
cd Environmental-Insights
python -m build
pip install dist/environmental_insights-0.2.1b0-py3-none-any.whl
```

---

## 📂 Data Sources

This package downloads and processes three primary CEDA datasets:

1. **Machine Learning for Hourly Air Pollution Prediction in England (ML-HAPPE)**  
   Berrisford, L. (2025). *Machine Learning for Hourly Air Pollution Prediction in England (ML-HAPPE).* NERC EDS Centre for Environmental Data Analysis.  
   DOI: [10.5285/fc735f9878ed43e293b85f85e40df24d](https://dx.doi.org/10.5285/fc735f9878ed43e293b85f85e40df24d)  
   > Full-year (2018) hourly modelled concentrations of NO₂, NO, NOₓ, O₃, PM₁₀, PM₂.₅ and SO₂ on a 1 km² grid, including 5th, 50th & 95th percentiles and underlying training data.

2. **Machine Learning for Hourly Air Pollution Prediction - Global (ML-HAPPG)**  
   Berrisford, L. (2025). *Machine Learning for Hourly Air Pollution Prediction – Global (ML-HAPPG).* NERC EDS Centre for Environmental Data Analysis.
   DOI: [10.5285/7f91b1326a324caa9e436b8fdef4a0d8](https://dx.doi.org/10.5285/7f91b1326a324caa9e436b8fdef4a0d8)  
   > Global hourly modelled concentrations for 2022 of NO₂, O₃, PM₁₀, PM₂.₅ and SO₂—offered on a 0.25° × 0.25° global grid with mean, 5th, 50th, and 95th percentile estimates.

3. **Synthetic Hourly Air Pollution Prediction Averages for England (SynthHAPPE)**  
   Berrisford, L. (2025). *Synthetic Hourly Air Pollution Prediction Averages for England (SynthHAPPE).* NERC EDS Centre for Environmental Data Analysis.  
   DOI: [10.5285/4cbd9c53ab07497ba42de5043d1f414b](https://dx.doi.org/10.5285/4cbd9c53ab07497ba42de5043d1f414b)  
   > Representative “typical day” profiles of NO₂, NO, NOₓ, O₃, PM₁₀, PM₂.₅ and SO₂ on a 1 km² grid, with 5th, 50th & 95th percentiles.



---

For full examples, see the Jupyter-Book tutorial in `book/tutorial_environmental_insights.ipynb`.

## 📚 Documentation

Build and view locally:

```bash
jupyter-book build book/
```

Then open `book/_build/html/index.html` in your browser.  
Highlights:

- **API Reference**: `book/docs/api/environmental_insights/`  
- **Tutorial Notebook**: `book/tutorial_environmental_insights.ipynb`

The documentation is also avaiable via the [GitHub Pages Site](https://liamjberrisford.github.io/Environmental-Insights/home_page.html)

---

## ✅ Testing

Run the full test suite:

```bash
pytest
```

Integration and unit tests are under `tests/`.

---

## 📑 Citation

If you use *Environmental Insights* in your work, please cite:

> Berrisford, L. J. (2025). Environmental Insights: Democratizing access to ambient air pollution data and predictive analytics (Version 0.2.1b0) [Software]. GitHub. https://github.com/liamjberrisford/Environmental-Insights  

Also cite the underlying datasets:

- Berrisford, L. (2025). *ML-HAPPE*: Machine Learning for Hourly Air Pollution Prediction in England. NERC EDS CEDA. DOI: 10.5285/fc735f9878ed43e293b85f85e40df24d
- Berrisford, L. (2025). *ML-HAPPG*: Machine Learning for Hourly Air Pollution Prediction - Global. NERC EDS CEDA. DOI: 10.5285/7f91b1326a324caa9e436b8fdef4a0d8 
- Berrisford, L. (2025). *SynthHAPPE*: Synthetic Hourly Air Pollution Prediction Averages for England. NERC EDS CEDA. DOI: 10.5285/4cbd9c53ab07497ba42de5043d1f414b  

---

## 📜 License

This project is released under the [GPL-3.0-or-later](LICENSE).  
