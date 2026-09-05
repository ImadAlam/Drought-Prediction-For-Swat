"""
Automated drought intelligence dashboard backend.

The application fetches live weather data from Open-Meteo, derives model inputs,
and exposes an automated dashboard payload for the frontend.
"""

import json
import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore")


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(BASE, "best_drought_model.pkl"))
SCALER = joblib.load(os.path.join(BASE, "best_scaler.pkl"))
FEATURES = joblib.load(os.path.join(BASE, "best_features.pkl"))
FEATURE_IDX = {feature: index for index, feature in enumerate(FEATURES)}

REFRESH_SECONDS = 30
CACHE_TTL_SECONDS = 25

LOCATIONS = {
    "swat": {
        "name": "Swat Valley",
        "lat": 35.2227,
        "lon": 72.4258,
        "alt": 980,
        "temp_normal_c": 18.0,
        "rain_normal_mm": 3.5,
        "soil_normal": 0.28,
        "ndvi_normal": 0.55,
    },
    "mingora": {
        "name": "Mingora",
        "lat": 34.7717,
        "lon": 72.3600,
        "alt": 890,
        "temp_normal_c": 22.0,
        "rain_normal_mm": 4.2,
        "soil_normal": 0.32,
        "ndvi_normal": 0.48,
    },
    "saidu": {
        "name": "Saidu Sharif",
        "lat": 34.7500,
        "lon": 72.3547,
        "alt": 905,
        "temp_normal_c": 22.0,
        "rain_normal_mm": 4.0,
        "soil_normal": 0.30,
        "ndvi_normal": 0.46,
    },
    "bahrain": {
        "name": "Bahrain",
        "lat": 35.2018,
        "lon": 72.5485,
        "alt": 1050,
        "temp_normal_c": 16.0,
        "rain_normal_mm": 5.1,
        "soil_normal": 0.35,
        "ndvi_normal": 0.62,
    },
    "kalam": {
        "name": "Kalam",
        "lat": 35.4904,
        "lon": 72.5773,
        "alt": 2100,
        "temp_normal_c": 10.0,
        "rain_normal_mm": 7.2,
        "soil_normal": 0.42,
        "ndvi_normal": 0.72,
    },
    "matta": {
        "name": "Matta",
        "lat": 34.9917,
        "lon": 72.3922,
        "alt": 920,
        "temp_normal_c": 20.0,
        "rain_normal_mm": 3.8,
        "soil_normal": 0.26,
        "ndvi_normal": 0.44,
    },
    "madyan": {
        "name": "Madyan",
        "lat": 35.1416,
        "lon": 72.5025,
        "alt": 1100,
        "temp_normal_c": 17.0,
        "rain_normal_mm": 4.8,
        "soil_normal": 0.33,
        "ndvi_normal": 0.58,
    },
    "chakdara": {
        "name": "Chakdara",
        "lat": 34.6840,
        "lon": 72.0509,
        "alt": 590,
        "temp_normal_c": 24.0,
        "rain_normal_mm": 2.9,
        "soil_normal": 0.22,
        "ndvi_normal": 0.38,
    },
    "khwazakhela": {
        "name": "Khwazakhela",
        "lat": 35.1000,
        "lon": 72.4100,
        "alt": 960,
        "temp_normal_c": 19.0,
        "rain_normal_mm": 4.0,
        "soil_normal": 0.29,
        "ndvi_normal": 0.52,
    },
}

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mostly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail",
}

dashboard_cache = {"timestamp": 0.0, "data": None}
cache_lock = Lock()

app = Flask(__name__, static_folder="static")


def utc_now():
    return datetime.now(timezone.utc)


def iso_utc(dt=None):
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value, low, high):
    return max(low, min(high, value))


def average(values, fallback=0.0):
    cleaned = [float(value) for value in values if value is not None]
    return float(sum(cleaned) / len(cleaned)) if cleaned else float(fallback)


def safe_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def series_tail(values, size, fallback):
    cleaned = [safe_float(value, fallback) for value in values if value is not None]
    if not cleaned:
        cleaned = [float(fallback)]
    while len(cleaned) < size:
        cleaned.insert(0, cleaned[0])
    return cleaned[-size:]


def model_level(probability):
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def ui_tier(probability):
    if probability < 0.25:
        return {"tier": "Minimal", "color": "#2f9e44"}
    if probability < 0.50:
        return {"tier": "Moderate", "color": "#7cb518"}
    if probability < 0.75:
        return {"tier": "Alert", "color": "#f08c00"}
    if probability < 0.90:
        return {"tier": "High", "color": "#d9480f"}
    return {"tier": "Emergency", "color": "#9c2a00"}


def weather_description(code):
    return WEATHER_CODES.get(int(code or 0), "Unspecified conditions")


def estimate_ndvi(temp_c, humidity_pct, soil_moisture, rain_mm, altitude_m, location_meta):
    moisture_factor = clamp((soil_moisture - 0.08) / 0.37, 0.0, 1.0)
    humidity_factor = clamp(humidity_pct / 100.0, 0.0, 1.0)
    rain_factor = clamp(rain_mm / 12.0, 0.0, 1.0)
    cooling_factor = clamp(1.0 - max(temp_c - 20.0, 0.0) / 18.0, 0.0, 1.0)
    altitude_factor = clamp(altitude_m / 2400.0, 0.0, 1.0)
    baseline = location_meta["ndvi_normal"] * 0.45
    ndvi = (
        0.12
        + baseline
        + 0.22 * moisture_factor
        + 0.08 * humidity_factor
        + 0.06 * rain_factor
        + 0.05 * cooling_factor
        + 0.03 * altitude_factor
    )
    return clamp(ndvi, 0.12, 0.88)


def build_feature_vector_from_series(temp_k_series, rain_m_series, soil_series, ndvi_series, month, evap_diff=0.0):
    temp_k = series_tail(temp_k_series, 6, temp_k_series[-1] if temp_k_series else 295.15)
    rain_m = series_tail(rain_m_series, 6, rain_m_series[-1] if rain_m_series else 0.001)
    soil = series_tail(soil_series, 6, soil_series[-1] if soil_series else 0.25)
    ndvi = series_tail(ndvi_series, 4, ndvi_series[-1] if ndvi_series else 0.45)

    vector = np.zeros(len(FEATURES))

    vector[FEATURE_IDX["t2m"]] = temp_k[-1]
    vector[FEATURE_IDX["t2m_lag_1"]] = temp_k[-2]
    vector[FEATURE_IDX["t2m_lag_2"]] = temp_k[-3]
    vector[FEATURE_IDX["t2m_lag_3"]] = temp_k[-4]
    vector[FEATURE_IDX["t2m_diff_1"]] = temp_k[-1] - temp_k[-2]

    vector[FEATURE_IDX["swvl1"]] = soil[-1]
    vector[FEATURE_IDX["swvl1_lag_1"]] = soil[-2]
    vector[FEATURE_IDX["swvl1_lag_2"]] = soil[-3]
    vector[FEATURE_IDX["swvl1_lag_3"]] = soil[-4]
    vector[FEATURE_IDX["swvl1_roll_mean_3"]] = average(soil[-3:], soil[-1])
    vector[FEATURE_IDX["swvl1_roll_mean_6"]] = average(soil, soil[-1])
    vector[FEATURE_IDX["swvl1_diff_1"]] = soil[-1] - soil[-2]

    vector[FEATURE_IDX["e_diff_1"]] = float(evap_diff)

    vector[FEATURE_IDX["tp"]] = rain_m[-1]
    vector[FEATURE_IDX["tp_lag_1"]] = rain_m[-2]
    vector[FEATURE_IDX["tp_lag_2"]] = rain_m[-3]
    vector[FEATURE_IDX["tp_lag_3"]] = rain_m[-4]
    vector[FEATURE_IDX["tp_roll_mean_3"]] = average(rain_m[-3:], rain_m[-1])
    vector[FEATURE_IDX["tp_roll_mean_6"]] = average(rain_m, rain_m[-1])
    vector[FEATURE_IDX["tp_diff_1"]] = rain_m[-1] - rain_m[-2]

    vector[FEATURE_IDX["mean_ndvi_lag_1"]] = ndvi[-2]
    vector[FEATURE_IDX["mean_ndvi_lag_3"]] = ndvi[-4]
    vector[FEATURE_IDX["mean_ndvi_diff_1"]] = ndvi[-1] - ndvi[-2]
    vector[FEATURE_IDX["month_cos"]] = math.cos(2 * math.pi * month / 12.0)

    return vector


def predict_risk(feature_vector):
    scaled = SCALER.transform(feature_vector.reshape(1, -1))
    probability = float(MODEL.predict_proba(scaled)[0][1])
    return probability


def build_open_meteo_url(location):
    params = {
        "latitude": location["lat"],
        "longitude": location["lon"],
        "timezone": "auto",
        "forecast_days": 16,
        "past_days": 3,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "rain",
                "weather_code",
                "wind_speed_10m",
            ]
        ),
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "soil_moisture_0_to_1cm",
            ]
        ),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "precipitation_probability_max",
                "relative_humidity_2m_mean",
                "wind_speed_10m_max",
            ]
        ),
    }
    return f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"


def fetch_json(url):
    request_obj = Request(url, headers={"User-Agent": "drought-dashboard/2.0"})
    with urlopen(request_obj, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def find_current_hour_index(hourly_times, current_time):
    try:
        return hourly_times.index(current_time)
    except ValueError:
        pass

    for index in range(len(hourly_times) - 1, -1, -1):
        if hourly_times[index] <= current_time:
            return index
    return max(0, len(hourly_times) - 1)


def group_hourly_by_date(times, values):
    grouped = {}
    for timestamp, value in zip(times, values):
        if value is None:
            continue
        date_key = timestamp.split("T")[0]
        grouped.setdefault(date_key, []).append(float(value))
    return grouped


def window_sum(entries, key, limit):
    return round(sum(float(entry.get(key, 0.0)) for entry in entries[:limit]), 2)


def fallback_snapshot(key, location, reason):
    now = utc_now()
    month = now.month
    seasonal_heat = 5.5 * math.sin(2 * math.pi * (month - 4) / 12.0)
    seasonal_rain = location["rain_normal_mm"] * (0.85 + 0.45 * math.cos(2 * math.pi * (month - 7) / 12.0))
    temp_c = round(location["temp_normal_c"] + seasonal_heat, 1)
    rain_mm = round(max(0.1, seasonal_rain), 2)
    humidity_pct = round(clamp(55 + 20 * math.cos(2 * math.pi * (month - 7) / 12.0), 28, 92), 1)
    soil_moisture = round(clamp(location["soil_normal"] * (0.92 + 0.15 * math.cos(2 * math.pi * month / 12.0)), 0.08, 0.55), 3)
    ndvi = round(estimate_ndvi(temp_c, humidity_pct, soil_moisture, rain_mm, location["alt"], location), 3)

    temp_k_series = [temp_c + 273.15] * 6
    rain_m_series = [rain_mm / 1000.0] * 6
    soil_series = [soil_moisture] * 6
    ndvi_series = [ndvi] * 4
    probability = predict_risk(
        build_feature_vector_from_series(temp_k_series, rain_m_series, soil_series, ndvi_series, month=month)
    )
    tier = ui_tier(probability)
    live_nowcast = []
    for hour_offset in range(12):
        hour_time = now + timedelta(hours=hour_offset)
        live_nowcast.append(
            {
                "time": iso_utc(hour_time),
                "label": hour_time.strftime("%H:%M"),
                "temperature_c": temp_c,
                "humidity_pct": humidity_pct,
                "rain_mm": rain_mm if hour_offset == 0 else round(rain_mm * 0.25, 2),
                "soil_moisture": soil_moisture,
                "ndvi": ndvi,
                "probability": round(probability, 4),
            }
        )

    daily_forecast = []
    for day_offset in range(7):
        day_time = now + timedelta(days=day_offset)
        daily_forecast.append(
            {
                "date": day_time.strftime("%Y-%m-%d"),
                "label": day_time.strftime("%d %b"),
                "temp_mean_c": temp_c,
                "temp_max_c": temp_c + 2,
                "temp_min_c": temp_c - 2,
                "rain_mm": rain_mm,
                "rain_probability_pct": 35.0,
                "soil_moisture": soil_moisture,
                "humidity_pct": humidity_pct,
                "ndvi": ndvi,
                "probability": round(probability, 4),
                "tier": tier["tier"],
                "color": tier["color"],
            }
        )

    outlook_windows = build_outlook_windows(
        location,
        {
            "temperature_c": temp_c,
            "rain_mm": rain_mm,
            "humidity_pct": humidity_pct,
            "soil_moisture": soil_moisture,
            "ndvi": ndvi,
            "probability": probability,
        },
        daily_forecast,
    )

    return {
        "key": key,
        "name": location["name"],
        "lat": location["lat"],
        "lon": location["lon"],
        "alt": location["alt"],
        "source": f"Fallback climate baseline ({reason})",
        "timestamp": iso_utc(now),
        "current": {
            "temperature_c": temp_c,
            "rain_mm": rain_mm,
            "humidity_pct": humidity_pct,
            "wind_ms": 4.5,
            "soil_moisture": soil_moisture,
            "ndvi": ndvi,
            "description": "Fallback baseline",
            "delta_1h": {
                "temperature_c": 0.0,
                "rain_mm": 0.0,
                "soil_moisture": 0.0,
                "ndvi": 0.0,
            },
        },
        "risk": {
            "probability": round(probability, 4),
            "model_level": model_level(probability),
            "tier": tier["tier"],
            "color": tier["color"],
        },
        "telemetry": {
            "fetch_duration_ms": 0,
            "next_6h_rain_mm": window_sum(live_nowcast, "rain_mm", 6),
            "next_12h_rain_mm": window_sum(live_nowcast, "rain_mm", 12),
            "peak_nowcast_probability": round(probability, 4),
        },
        "live_nowcast": live_nowcast,
        "daily_forecast": daily_forecast,
        "outlook_windows": outlook_windows,
        "drivers": [
            {"label": "Data source", "value": "Fallback baseline", "impact": "medium"},
            {"label": "Temperature", "value": f"{temp_c:.1f} C seasonal baseline", "impact": "medium"},
            {"label": "Soil moisture", "value": f"{soil_moisture:.2f} m3/m3", "impact": "medium"},
        ],
        "alerts": [
            {
                "level": "warning",
                "title": "Live feed unavailable",
                "detail": "Seasonal fallback values are being used until the live weather request succeeds.",
            }
        ],
        "activity": [
            {
                "time": now.strftime("%H:%M UTC"),
                "title": "Live fetch unavailable",
                "detail": reason,
                "tone": "warn",
            }
        ],
    }


def build_outlook_windows(location, current, forecast_days):
    windows = []
    anchor = utc_now()
    if forecast_days:
        mean_risk = average([day["probability"] for day in forecast_days[:7]], current["probability"])
        mean_rain = average([day["rain_mm"] for day in forecast_days[:7]], current["rain_mm"])
        mean_temp = average([day["temp_mean_c"] for day in forecast_days[:7]], current["temperature_c"])
        mean_soil = average([day["soil_moisture"] for day in forecast_days[:7]], current["soil_moisture"])
        mean_ndvi = average([day["ndvi"] for day in forecast_days[:7]], current["ndvi"])
    else:
        mean_risk = current["probability"]
        mean_rain = current["rain_mm"]
        mean_temp = current["temperature_c"]
        mean_soil = current["soil_moisture"]
        mean_ndvi = current["ndvi"]

    windows.append(
        {
            "label": "7-Day",
            "days_ahead": 7,
            "probability": round(mean_risk, 4),
            "tier": ui_tier(mean_risk)["tier"],
            "color": ui_tier(mean_risk)["color"],
            "temp_c": round(mean_temp, 1),
            "rain_mm": round(mean_rain, 1),
            "soil_moisture": round(mean_soil, 3),
            "ndvi": round(mean_ndvi, 3),
            "confidence": "High",
        }
    )

    for days_ahead, confidence in ((30, "Medium"), (90, "Medium")):
        target = anchor + timedelta(days=days_ahead)
        month_shift = math.sin(2 * math.pi * (target.month - 4) / 12.0) - math.sin(
            2 * math.pi * (anchor.month - 4) / 12.0
        )
        rain_season = 0.85 + 0.45 * math.cos(2 * math.pi * (target.month - 7) / 12.0)

        temp_c = current["temperature_c"] + month_shift * 5.0
        rain_mm = max(0.1, mean_rain * 0.45 + location["rain_normal_mm"] * rain_season)
        soil_moisture = clamp(
            mean_soil * 0.78 + location["soil_normal"] * 0.35 - max(temp_c - current["temperature_c"], 0) * 0.004,
            0.06,
            0.55,
        )
        humidity_pct = clamp(
            current["humidity_pct"] * 0.62 + 50.0 + 12.0 * math.cos(2 * math.pi * (target.month - 7) / 12.0),
            20.0,
            96.0,
        )
        ndvi = estimate_ndvi(temp_c, humidity_pct, soil_moisture, rain_mm, location["alt"], location)

        probability = predict_risk(
            build_feature_vector_from_series(
                [temp_c + 273.15] * 6,
                [rain_mm / 1000.0] * 6,
                [soil_moisture] * 6,
                [ndvi] * 4,
                month=target.month,
            )
        )
        tier = ui_tier(probability)
        windows.append(
            {
                "label": f"{days_ahead}-Day",
                "days_ahead": days_ahead,
                "probability": round(probability, 4),
                "tier": tier["tier"],
                "color": tier["color"],
                "temp_c": round(temp_c, 1),
                "rain_mm": round(rain_mm, 1),
                "soil_moisture": round(soil_moisture, 3),
                "ndvi": round(ndvi, 3),
                "confidence": confidence,
            }
        )

    return windows


def build_location_snapshot(key, location):
    try:
        fetch_started = time.perf_counter()
        payload = fetch_json(build_open_meteo_url(location))
        current = payload["current"]
        hourly = payload["hourly"]
        daily = payload["daily"]

        current_time = current["time"]
        hourly_times = hourly["time"]
        current_index = find_current_hour_index(hourly_times, current_time)

        temperature_series_c = series_tail(hourly["temperature_2m"][: current_index + 1], 6, location["temp_normal_c"])
        humidity_series = series_tail(
            hourly["relative_humidity_2m"][: current_index + 1], 6, 55.0
        )
        rain_series_mm = series_tail(hourly["precipitation"][: current_index + 1], 6, location["rain_normal_mm"])
        soil_series = series_tail(
            hourly["soil_moisture_0_to_1cm"][: current_index + 1], 6, location["soil_normal"]
        )

        current_temp_c = safe_float(current.get("temperature_2m"), temperature_series_c[-1])
        current_humidity = safe_float(current.get("relative_humidity_2m"), humidity_series[-1])
        current_rain_mm = safe_float(current.get("rain"), rain_series_mm[-1])
        current_soil = soil_series[-1]
        previous_temp_c = temperature_series_c[-2]
        previous_humidity = humidity_series[-2]
        previous_rain_mm = rain_series_mm[-2]
        previous_soil = soil_series[-2]
        current_ndvi = estimate_ndvi(
            current_temp_c,
            current_humidity,
            current_soil,
            current_rain_mm,
            location["alt"],
            location,
        )
        previous_ndvi = estimate_ndvi(
            previous_temp_c,
            previous_humidity,
            previous_soil,
            previous_rain_mm,
            location["alt"],
            location,
        )

        ndvi_history = [
            estimate_ndvi(temp, humidity, soil, rain, location["alt"], location)
            for temp, humidity, soil, rain in zip(temperature_series_c, humidity_series, soil_series, rain_series_mm)
        ]

        probability = predict_risk(
            build_feature_vector_from_series(
                [value + 273.15 for value in temperature_series_c],
                [value / 1000.0 for value in rain_series_mm],
                soil_series,
                ndvi_history[-4:],
                month=utc_now().month,
            )
        )
        tier = ui_tier(probability)

        grouped_soil = group_hourly_by_date(hourly_times, hourly["soil_moisture_0_to_1cm"])
        forecast_days = []
        live_nowcast = []

        path_temp_series = list(temperature_series_c)
        path_rain_series = list(rain_series_mm)
        path_soil_series = list(soil_series)
        path_ndvi_series = list(ndvi_history[-4:])

        for index in range(current_index, min(current_index + 12, len(hourly_times))):
            hour_label = hourly_times[index].split("T")[-1][:5]
            hour_month = int(hourly_times[index][5:7])
            temp_c = safe_float(hourly["temperature_2m"][index], current_temp_c)
            humidity_pct = safe_float(hourly["relative_humidity_2m"][index], current_humidity)
            rain_mm = safe_float(hourly["precipitation"][index], 0.0)
            soil_value = safe_float(hourly["soil_moisture_0_to_1cm"][index], current_soil)
            ndvi_value = round(
                estimate_ndvi(temp_c, humidity_pct, soil_value, rain_mm, location["alt"], location),
                3,
            )

            path_temp_series.append(temp_c)
            path_rain_series.append(rain_mm)
            path_soil_series.append(soil_value)
            path_ndvi_series.append(ndvi_value)

            hourly_probability = predict_risk(
                build_feature_vector_from_series(
                    [value + 273.15 for value in path_temp_series[-6:]],
                    [value / 1000.0 for value in path_rain_series[-6:]],
                    path_soil_series[-6:],
                    path_ndvi_series[-4:],
                    month=hour_month,
                )
            )

            live_nowcast.append(
                {
                    "time": hourly_times[index],
                    "label": hour_label,
                    "temperature_c": round(temp_c, 1),
                    "humidity_pct": round(humidity_pct, 1),
                    "rain_mm": round(rain_mm, 2),
                    "soil_moisture": round(soil_value, 3),
                    "ndvi": ndvi_value,
                    "probability": round(hourly_probability, 4),
                }
            )

        rolling_temp = [value + 273.15 for value in temperature_series_c]
        rolling_rain = [value / 1000.0 for value in rain_series_mm]
        rolling_soil = list(soil_series)
        rolling_ndvi = list(ndvi_history[-4:])

        daily_start_index = 0
        current_date = current_time.split("T")[0]
        for index, day in enumerate(daily["time"]):
            if day >= current_date:
                daily_start_index = index
                break

        for index in range(daily_start_index, min(daily_start_index + 7, len(daily["time"]))):
            day = daily["time"][index]
            mean_temp_c = safe_float(daily["temperature_2m_mean"][index], current_temp_c)
            rain_mm = safe_float(daily["precipitation_sum"][index], current_rain_mm)
            humidity_pct = safe_float(daily["relative_humidity_2m_mean"][index], current_humidity)
            soil_moisture = round(average(grouped_soil.get(day, []), current_soil), 3)
            ndvi = round(
                estimate_ndvi(mean_temp_c, humidity_pct, soil_moisture, rain_mm, location["alt"], location),
                3,
            )

            rolling_temp.append(mean_temp_c + 273.15)
            rolling_rain.append(rain_mm / 1000.0)
            rolling_soil.append(soil_moisture)
            rolling_ndvi.append(ndvi)

            forecast_probability = predict_risk(
                build_feature_vector_from_series(
                    rolling_temp[-6:],
                    rolling_rain[-6:],
                    rolling_soil[-6:],
                    rolling_ndvi[-4:],
                    month=datetime.fromisoformat(day).month,
                )
            )
            forecast_tier = ui_tier(forecast_probability)

            forecast_days.append(
                {
                    "date": day,
                    "label": datetime.fromisoformat(day).strftime("%d %b"),
                    "temp_mean_c": round(mean_temp_c, 1),
                    "temp_max_c": round(safe_float(daily["temperature_2m_max"][index], mean_temp_c), 1),
                    "temp_min_c": round(safe_float(daily["temperature_2m_min"][index], mean_temp_c), 1),
                    "rain_mm": round(rain_mm, 1),
                    "rain_probability_pct": round(
                        safe_float(daily["precipitation_probability_max"][index], 0.0), 1
                    ),
                    "soil_moisture": soil_moisture,
                    "humidity_pct": round(humidity_pct, 1),
                    "ndvi": ndvi,
                    "probability": round(forecast_probability, 4),
                    "tier": forecast_tier["tier"],
                    "color": forecast_tier["color"],
                }
            )

        temp_delta = current_temp_c - location["temp_normal_c"]
        rain_ratio = current_rain_mm / max(location["rain_normal_mm"], 0.1)
        soil_delta = current_soil - location["soil_normal"]
        ndvi_delta = current_ndvi - location["ndvi_normal"]

        activity = []
        if rain_ratio < 0.75:
            activity.append(
                {
                    "time": current_time.split("T")[-1],
                    "title": "Rainfall deficit flagged",
                    "detail": f"Observed rain is {round((1 - rain_ratio) * 100)}% below local baseline.",
                    "tone": "warn",
                }
            )
        else:
            activity.append(
                {
                    "time": current_time.split("T")[-1],
                    "title": "Moisture recharge detected",
                    "detail": "Recent rainfall is supporting short-term soil recovery.",
                    "tone": "good",
                }
            )

        activity.append(
            {
                "time": current_time.split("T")[-1],
                "title": "Vegetation signal updated",
                "detail": f"NDVI proxy is {current_ndvi:.2f} against a local normal of {location['ndvi_normal']:.2f}.",
                "tone": "info",
            }
        )

        activity.append(
            {
                "time": current_time.split("T")[-1],
                "title": "AI forecast refreshed",
                "detail": "7-day risk curve rebuilt from live weather and forecast data.",
                "tone": "info",
            }
        )

        drivers = [
            {
                "label": "Temperature anomaly",
                "value": f"{temp_delta:+.1f} C vs normal",
                "impact": "high" if temp_delta > 4 else "medium" if temp_delta > 1.5 else "low",
            },
            {
                "label": "Rain status",
                "value": f"{current_rain_mm:.1f} mm now, baseline {location['rain_normal_mm']:.1f} mm",
                "impact": "high" if rain_ratio < 0.6 else "medium" if rain_ratio < 0.9 else "low",
            },
            {
                "label": "Soil moisture",
                "value": f"{current_soil:.3f} m3/m3",
                "impact": "high" if soil_delta < -0.08 else "medium" if soil_delta < -0.03 else "low",
            },
            {
                "label": "Vegetation proxy",
                "value": f"{current_ndvi:.2f} NDVI equivalent",
                "impact": "high" if ndvi_delta < -0.12 else "medium" if ndvi_delta < -0.05 else "low",
            },
        ]

        alerts = []
        if probability >= 0.85:
            alerts.append(
                {
                    "level": "critical",
                    "title": "Critical drought pressure",
                    "detail": "Current model risk is above 85% and needs immediate monitoring.",
                }
            )
        elif probability >= 0.70:
            alerts.append(
                {
                    "level": "warning",
                    "title": "Elevated drought watch",
                    "detail": "Current model risk is above 70% with short-term stress signals present.",
                }
            )

        if live_nowcast and window_sum(live_nowcast, "rain_mm", 6) < 1.0:
            alerts.append(
                {
                    "level": "warning",
                    "title": "Weak 6-hour rainfall outlook",
                    "detail": "The next 6 hours show little recharge potential from rainfall.",
                }
            )

        if current_soil < 0.20:
            alerts.append(
                {
                    "level": "warning",
                    "title": "Dry surface soil",
                    "detail": "Surface soil moisture is below 0.20 m3/m3.",
                }
            )

        if not alerts:
            alerts.append(
                {
                    "level": "ok",
                    "title": "System stable",
                    "detail": "No critical short-horizon drought alert is active right now.",
                }
            )

        fetch_duration_ms = int((time.perf_counter() - fetch_started) * 1000)

        snapshot = {
            "key": key,
            "name": location["name"],
            "lat": location["lat"],
            "lon": location["lon"],
            "alt": location["alt"],
            "source": "Open-Meteo forecast API",
            "timestamp": iso_utc(),
            "current": {
                "temperature_c": round(current_temp_c, 1),
                "rain_mm": round(current_rain_mm, 2),
                "humidity_pct": round(current_humidity, 1),
                "wind_ms": round(safe_float(current.get("wind_speed_10m"), 0.0), 1),
                "soil_moisture": round(current_soil, 3),
                "ndvi": round(current_ndvi, 3),
                "description": weather_description(current.get("weather_code")),
                "delta_1h": {
                    "temperature_c": round(current_temp_c - previous_temp_c, 1),
                    "rain_mm": round(current_rain_mm - previous_rain_mm, 2),
                    "soil_moisture": round(current_soil - previous_soil, 3),
                    "ndvi": round(current_ndvi - previous_ndvi, 3),
                },
            },
            "risk": {
                "probability": round(probability, 4),
                "model_level": model_level(probability),
                "tier": tier["tier"],
                "color": tier["color"],
            },
            "telemetry": {
                "fetch_duration_ms": fetch_duration_ms,
                "next_6h_rain_mm": window_sum(live_nowcast, "rain_mm", 6),
                "next_12h_rain_mm": window_sum(live_nowcast, "rain_mm", 12),
                "peak_nowcast_probability": round(
                    max([entry["probability"] for entry in live_nowcast], default=probability), 4
                ),
            },
            "live_nowcast": live_nowcast,
            "daily_forecast": forecast_days,
            "outlook_windows": build_outlook_windows(
                location,
                {
                    "temperature_c": current_temp_c,
                    "rain_mm": current_rain_mm,
                    "humidity_pct": current_humidity,
                    "soil_moisture": current_soil,
                    "ndvi": current_ndvi,
                    "probability": probability,
                },
                forecast_days,
            ),
            "drivers": drivers,
            "alerts": alerts,
            "activity": activity,
        }
        return snapshot
    except Exception as exc:
        return fallback_snapshot(key, location, str(exc))


def build_grid(selected_key, snapshots):
    grid = []
    selected = snapshots[selected_key]
    grid_min_lat, grid_max_lat = 34.50, 36.00
    grid_min_lon, grid_max_lon = 71.80, 73.20
    rows, cols = 8, 8
    delta_lat = (grid_max_lat - grid_min_lat) / rows
    delta_lon = (grid_max_lon - grid_min_lon) / cols

    def in_region(lat, lon):
        center_lat, center_lon = 35.25, 72.40
        lat_radius, lon_radius = 0.85, 0.75
        dx = (lon - center_lon) / lon_radius
        dy = (lat - center_lat) / lat_radius
        return dx * dx + dy * dy <= 1.65

    for row in range(rows):
        for col in range(cols):
            lat0 = grid_min_lat + row * delta_lat
            lon0 = grid_min_lon + col * delta_lon
            lat1 = lat0 + delta_lat
            lon1 = lon0 + delta_lon
            cell_lat = (lat0 + lat1) / 2.0
            cell_lon = (lon0 + lon1) / 2.0
            if not in_region(cell_lat, cell_lon):
                continue

            weighted_sum = 0.0
            weight_total = 0.0
            for item in snapshots.values():
                dist = math.sqrt((cell_lat - item["lat"]) ** 2 + (cell_lon - item["lon"]) ** 2)
                weight = 1.0 / max(0.03, dist)
                weighted_sum += item["risk"]["probability"] * weight
                weight_total += weight

            base_probability = weighted_sum / max(weight_total, 1e-6)
            terrain_shift = 0.04 * math.sin((cell_lon - 71.8) * 3.1) - 0.05 * max(0.0, (cell_lat - 34.7) / 1.3)
            selected_bias = (selected["risk"]["probability"] - 0.5) * 0.08
            probability = clamp(base_probability + terrain_shift + selected_bias, 0.03, 0.97)
            tier = ui_tier(probability)

            grid.append(
                {
                    "bounds": [[round(lat0, 4), round(lon0, 4)], [round(lat1, 4), round(lon1, 4)]],
                    "probability": round(probability, 4),
                    "tier": tier["tier"],
                    "color": tier["color"],
                }
            )
    return grid


def build_dashboard_payload(selected_key):
    selected_key = selected_key if selected_key in LOCATIONS else "swat"
    snapshots = {}
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(6, len(LOCATIONS))) as executor:
        future_to_key = {
            executor.submit(build_location_snapshot, key, location): key for key, location in LOCATIONS.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            snapshots[key] = future.result()

    ordered_locations = sorted(
        snapshots.values(), key=lambda item: item["risk"]["probability"], reverse=True
    )
    selected = snapshots[selected_key]
    hotspots = ordered_locations[:4]
    generated_at = iso_utc()
    average_risk = average([item["risk"]["probability"] for item in ordered_locations], selected["risk"]["probability"])
    high_risk_count = sum(1 for item in ordered_locations if item["risk"]["probability"] >= 0.75)
    alert_count = sum(1 for item in ordered_locations for alert in item.get("alerts", []) if alert["level"] in {"warning", "critical"})
    hottest = max(ordered_locations, key=lambda item: item["current"]["temperature_c"])
    driest = min(ordered_locations, key=lambda item: item["current"]["soil_moisture"])
    wettest_window = max(ordered_locations, key=lambda item: item["telemetry"]["next_12h_rain_mm"])
    build_duration_ms = int((time.perf_counter() - started) * 1000)

    insights = [
        {
            "title": "Auto AI mode is active",
            "detail": "Manual override has been removed. Predictions now refresh from live weather and forecast inputs every 30 seconds.",
        },
        {
            "title": "Current lead driver",
            "detail": f"{selected['drivers'][0]['label']}: {selected['drivers'][0]['value']}.",
        },
        {
            "title": "Regional watch",
            "detail": f"Highest current risk is {hotspots[0]['name']} at {round(hotspots[0]['risk']['probability'] * 100)}%.",
        },
    ]

    regional_summary = {
        "average_risk": round(average_risk, 4),
        "high_risk_count": high_risk_count,
        "alert_count": alert_count,
        "build_duration_ms": build_duration_ms,
        "hottest_location": {
            "name": hottest["name"],
            "temperature_c": hottest["current"]["temperature_c"],
        },
        "driest_location": {
            "name": driest["name"],
            "soil_moisture": driest["current"]["soil_moisture"],
        },
        "wettest_window_location": {
            "name": wettest_window["name"],
            "rain_mm": wettest_window["telemetry"]["next_12h_rain_mm"],
        },
    }

    return {
        "generated_at": generated_at,
        "refresh_interval_seconds": REFRESH_SECONDS,
        "source": "Open-Meteo live weather + AI drought model",
        "regional_summary": regional_summary,
        "selected_location": selected,
        "locations": ordered_locations,
        "hotspots": hotspots,
        "grid": build_grid(selected_key, snapshots),
        "insights": insights,
    }


def get_dashboard_payload(selected_key):
    now = time.time()
    with cache_lock:
        cached = dashboard_cache["data"]
        if cached and now - dashboard_cache["timestamp"] < CACHE_TTL_SECONDS:
            if selected_key == cached["selected_location"]["key"]:
                return cached

    payload = build_dashboard_payload(selected_key)
    with cache_lock:
        dashboard_cache["timestamp"] = now
        dashboard_cache["data"] = payload
    return payload


@app.route("/")
def index():
    return send_from_directory(BASE, "dashboard.html")


@app.route("/dashboard-data")
def dashboard_data():
    selected_key = request.args.get("location", "swat").strip().lower()
    return jsonify(get_dashboard_payload(selected_key))


@app.route("/forecast")
def forecast():
    selected_key = request.args.get("location", "swat").strip().lower()
    payload = get_dashboard_payload(selected_key)
    selected = payload["selected_location"]
    return jsonify(
        {
            "location": selected["name"],
            "location_key": selected["key"],
            "generated_at": payload["generated_at"],
            "daily_forecast": selected["daily_forecast"],
            "outlook_windows": selected["outlook_windows"],
        }
    )


@app.route("/live-weather")
def live_weather():
    selected_key = request.args.get("location", "swat").strip().lower()
    payload = get_dashboard_payload(selected_key)
    selected = payload["selected_location"]
    return jsonify(
        {
            "location": {"key": selected["key"], "name": selected["name"], "lat": selected["lat"], "lon": selected["lon"]},
            "source": selected["source"],
            "timestamp": selected["timestamp"],
            "current": selected["current"],
            "risk": selected["risk"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        temperature_c = safe_float(data.get("temperature_c", data.get("temperature", 22.0)), 22.0)
        rain_mm = safe_float(data.get("rain_mm", data.get("precipitation_mm", 3.0)), 3.0)
        soil_moisture = safe_float(data.get("soil_moisture", 0.28), 0.28)
        ndvi = safe_float(data.get("ndvi", 0.45), 0.45)
        month = int(safe_float(data.get("month", utc_now().month), utc_now().month))

        probability = predict_risk(
            build_feature_vector_from_series(
                [temperature_c + 273.15] * 6,
                [rain_mm / 1000.0] * 6,
                [soil_moisture] * 6,
                [ndvi] * 4,
                month=month,
            )
        )
        tier = ui_tier(probability)

        return jsonify(
            {
                "probability": round(probability, 4),
                "risk_level": model_level(probability),
                "tier": tier["tier"],
                "risk_color": tier["color"],
                "timestamp": iso_utc(),
                "automation_note": "Manual UI override was removed, but this endpoint remains available for diagnostics.",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("=" * 60)
    print("  Automated Drought Intelligence Dashboard")
    print(f"  Model: RandomForestClassifier ({len(FEATURES)} features)")
    print("  Live source: Open-Meteo forecast API")
    print("  Open: http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)
