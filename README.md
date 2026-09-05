# Drought Intelligence 

## A Research-Driven Early Warning Project for Swat Valley

An applied machine learning project for monitoring drought risk across Swat Valley and nearby districts in Khyber Pakhtunkhwa, Pakistan. The system combines live weather data from the Open-Meteo forecast API, engineered hydroclimate features, a trained drought classifier, and a Flask-based dashboard with prediction endpoints.

> **Research focus:** Automated drought-risk monitoring for mountainous districts.
>
> **Forecast view:** Live conditions, a 7-day forecast, and 7/30/90-day outlook windows.
>
> **Study region:** Swat Valley and nearby locations in Khyber Pakhtunkhwa, Pakistan.

## About the Research Project

Drought conditions develop through interacting changes in temperature, precipitation, soil moisture, evaporation, and vegetation health. Mountainous regions are especially sensitive to these changes because elevation, local climate, and water availability vary significantly between nearby locations.

This project turns those signals into an accessible monitoring interface. The backend retrieves current and forecast weather data, derives model inputs, estimates a drought probability for each monitored location, and supplies the dashboard with regional risk summaries, hotspot rankings, alerts, and outlooks.

The application is designed for research, experimentation, and decision-support development. It is not a replacement for official meteorological or emergency-management warnings.

## Objectives

- Monitor drought risk automatically across selected Swat Valley locations.
- Use live weather observations and forecasts from the Open-Meteo API.
- Apply a trained machine learning classifier to engineered hydroclimate features.
- Provide current risk, regional comparisons, hotspot rankings, and forecast outlooks.
- Present the results through a browser-based Flask dashboard.
- Expose JSON endpoints so the monitoring output can be integrated with other systems.
- Continue operating with a seasonal baseline when live weather data cannot be fetched.

## Methodology

### Data and Features

For each monitored location, the application combines live weather variables with location metadata such as elevation, normal temperature, normal rainfall, typical soil moisture, and vegetation baseline. The backend derives time-series features including:

- Temperature and temperature lag values.
- Soil-moisture levels, rolling means, and changes.
- Precipitation levels, rolling means, and changes.
- Evaporation-related differences.
- Estimated vegetation condition using an NDVI proxy.
- Seasonal month features.

The current weather source is the Open-Meteo forecast API. When a live request fails, the backend uses a seasonal fallback so the dashboard can still display a usable monitoring state.

### Model and Risk Scoring

The deployed model is loaded from `best_drought_model.pkl`. Input values are transformed with `best_scaler.pkl`, and the expected feature order is loaded from `best_features.pkl`.

The deployed classifier is a balanced Random Forest with 150 trees, a maximum depth of 6, and random state 42. It receives 24 standardized features covering temperature, soil moisture, precipitation, evaporation difference, vegetation proxies, lag values, rolling means, first differences, and seasonal information.

The classifier produces a drought probability. The application converts that probability into model levels and interface tiers such as Minimal, Moderate, Alert, High, and Emergency. The dashboard also derives regional summaries, map risk grids, active alerts, and future outlook windows from the location-level results.

### Application Architecture

1. Open-Meteo supplies current and forecast weather data.
2. Flask fetches and normalizes the weather response.
3. Feature engineering builds the model input vector for each location.
4. The scaler and trained classifier generate drought probabilities.
5. Flask returns dashboard-ready JSON data.
6. `dashboard.html` renders the live monitoring interface in a browser.

### End-to-End Processing Workflow

The model and dashboard follow this sequence:

1. **Collect inputs:** Retrieve current and forecast weather variables for each configured latitude and longitude from Open-Meteo.
2. **Normalize units:** Convert temperature and precipitation values into the units expected by the trained feature pipeline.
3. **Build temporal features:** Construct recent lags, rolling averages, changes, and the cyclical month feature from the weather series.
4. **Estimate vegetation condition:** Derive an NDVI-style vegetation proxy from moisture, humidity, rainfall, temperature, elevation, and the location baseline.
5. **Arrange feature order:** Place all values according to `best_features.pkl`, ensuring inference uses the same 24-column order as training.
6. **Scale inputs:** Apply the saved `StandardScaler` from `best_scaler.pkl`.
7. **Predict probability:** Use the trained Random Forest to calculate the probability of drought risk.
8. **Assign risk tier:** Convert the probability into a model level and a dashboard tier for user-facing interpretation.
9. **Aggregate results:** Combine location scores into rankings, alerts, regional summaries, forecast outlooks, and map layers.
10. **Serve the interface:** Return the dashboard payload through Flask and render it in `dashboard.html`.

## Model Performance and Results

The model comparison results are stored in `Simple_Drought_Model_Table.xlsx`. The evaluation compares five classification algorithms using balanced accuracy, test accuracy, precision, recall, F1 score, AUC, cross-validation score, and the train-test balanced-accuracy gap.

| Rank | Model | Train Bal. Acc. | Test Bal. Acc. | Test Acc. | Precision | Recall | F1 Score | AUC | CV Score | Overfit Gap | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Random Forest | 98.22% | 89.35% | 90.00% | 87.50% | 97.22% | 92.11% | 0.9588 | 0.8119 | 8.87% | Best overall; deployed |
| 2 | Logistic Regression | 85.99% | 88.43% | 89.00% | 89.19% | 91.67% | 90.41% | 0.9362 | 0.8104 | -2.44% | Most stable |
| 3 | XGBoost | 97.15% | 87.04% | 87.00% | 88.89% | 88.89% | 88.89% | 0.9660 | 0.8080 | 10.11% | Highest AUC |
| 4 | Decision Tree | 89.42% | 86.57% | 86.00% | 93.55% | 80.56% | 86.57% | 0.9043 | 0.7389 | 2.84% | Highest precision |
| 5 | SVM | 86.68% | 84.26% | 86.00% | 82.93% | 94.44% | 88.31% | 0.9434 | 0.7647 | 2.42% | High recall |

### Interpretation of Results

Random Forest was selected as the deployed model because it achieved the strongest overall balance across the reported metrics. Its 97.22% recall means it identified most of the positive drought-risk cases in the test results, while its 87.50% precision limited the number of risk alerts that were false positives. Its 92.11% F1 score summarizes this precision-recall balance.

XGBoost achieved the highest AUC at 0.9660, indicating strong ranking ability across thresholds, but its test accuracy, recall, and F1 score were lower than the deployed Random Forest in the comparison table. Accuracy should therefore be read together with recall, precision, F1, and AUC when assessing an early-warning model.

The 8.87% Random Forest train-test balanced-accuracy gap indicates some difference between training and test performance. The model should be re-evaluated on new historical periods and independent local observations before operational deployment.

## Dashboard Capabilities

- Automated refresh designed around a 30-second data loop.
- Current drought-risk snapshot for the selected location.
- Monitoring across Swat Valley and nearby districts.
- 12-hour live nowcast and 7-day forecast series.
- 7-day, 30-day, and 90-day outlook windows.
- Regional pulse summaries and hotspot ranking.
- Risk map grid and active alerts.
- One-hour metric change tracking.
- AI-estimated vegetation proxy based on live hydroclimate inputs.

## Project Structure

```text
drought_app/
├── README.md
├── app.py
├── dashboard.html
├── requirements.txt
├── best_drought_model.pkl
├── best_scaler.pkl
├── best_features.pkl
├── generate_drought_thesis.py
├── Simple_Drought_Model_Table.xlsx
├── start.sh
├── .gitignore
└── generated_thesis_assets/
```

The three `.pkl` files are required at runtime because the Flask application loads them when it starts. The `generated_thesis_assets/` folder contains supporting figures. The local `.venv/` and `__pycache__/` folders are development files and should not be uploaded.

## How to Use

### 1. Install dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

### 2. Check the model files

Confirm that these files are in the project root:

```text
best_drought_model.pkl
best_scaler.pkl
best_features.pkl
```

### 3. Start the application

```bash
python app.py
```

The server runs on port `5050`. Open:

```text
http://127.0.0.1:5050
```

On Linux or macOS, the helper script can also be used:

```bash
bash start.sh
```

## API Endpoints

### `GET /dashboard-data?location=swat`

Returns the complete dashboard payload, including the selected location, monitored locations, forecast series, outlook windows, activity feed, AI insights, and map risk grid.

### `GET /live-weather?location=swat`

Returns current weather conditions and the current drought score for one location.

### `GET /forecast?location=swat`

Returns the selected location's 7-day forecast and 7/30/90-day outlook windows.

### `POST /predict`

Provides a diagnostic prediction endpoint for direct input values. The current dashboard uses automated live-data scoring instead of a manual slider interface.

## Limitations and Future Work

- Open-Meteo forecast data and the model inputs may not capture highly local rainfall or soil conditions.
- The NDVI value displayed by the dashboard is an estimated vegetation proxy, not a direct satellite observation.
- Fallback values keep the interface functional but should not be treated as live measurements.
- The current monitored region and baseline values are configured in `app.py`.
- Future work can add satellite vegetation products, local sensor data, uncertainty estimates, historical evaluation reports, user-configurable alerts, and deployment monitoring.

## Conclusion

This project demonstrates a practical connection between machine learning, live weather retrieval, and an accessible drought early-warning interface. Its main contribution is a runnable monitoring workflow that turns hydroclimate signals into location-level risk information for a mountainous region.

The system is intended for research, experimentation, and decision-support development. Official meteorological warnings and local emergency procedures should remain the authority for operational decisions.



## Author

**Imad Alam**

## Stay Updated and Join the Community

For research updates, collaboration, and discussion:

- Email: [alam1122imad@gmail.com](mailto:alam1122imad@gmail.com)
- LinkedIn: [Imad Alam](https://www.linkedin.com/in/imad-alam-85b4aa25a)

Contributions, research feedback, reproducibility improvements, and ideas for higher-resolution drought monitoring are welcome.
