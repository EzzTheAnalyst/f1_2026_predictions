# importing needed libraries
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import requests


"""
Barcelona GP 2026 - Winner Prediction Model
===========================================


Approach: Train on Free Practice (FP1,FP2,FP3,Q)
Reasoning:
    - Qualifying: Best single-lap speed signal
    - FP: race-pace
    - Race weather (weather has a big effect on the tires and time race)
    - using both gives context to the model

"""

fp_sessions = {}
for session_name in ["FP2", "FP3", "Q"]:
    session = fastf1.get_session(2026, "Barcelona", session_name)
    session.load()


    # get each driver's personal best lap in the fp
    fastest = (session.laps.pick_quicklaps().groupby("Driver")["LapTime"].min().reset_index().rename(columns={"LapTime":session_name}))
    fastest[session_name] = fastest[session_name].dt.total_seconds()
    fp_sessions[session_name] = fastest


# merge all FP on driver
df_fp = fp_sessions["FP2"]
for i in ["FP3", "Q"]:
    df_fp = df_fp.merge(fp_sessions[i], on="Driver", how="outer")

# mean fastest lap across all 3 sessions with ignoring na values in case a driver missed a session
df_fp["fp_mean_best_lap"] = df_fp[["FP2","FP3","Q"]].mean(axis=1)

# to avoid training with NaN values
df_fp.dropna(inplace=True)


# creating a dataframe with the qualified people only and their best lap time
qualifying_2026 = pd.DataFrame({"Driver":

["ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "COL", "GAS", "HAD", "HAM", "HUL", "LAW", "LEC", "LIN", "NOR", "OCO", "PER", "PIA", "RUS", "SAI", "STR", "VER"],

"QualifyingTime (s)": [ 77.424,     #ALB
                        78.815,     #ALO
                        74.998,     #ANT
                        76.389,     #BEA
                        76.001,     #BOR
                        77.757,     #BOT
                        76.191,     #COL
                        76.261,     #GAS
                        75.077,     #HAD
                        74.743,     #HAM
                        76.657,     #HUL
                        76.542,     #LAW
                        75.281,     #LEC
                        75.840,     #LIN
                        75.001,     #NOR
                        77.073,     #OCO
                        77.545,     #PER
                        75.090,     #PIA
                        74.679,     #RUS
                        77.827,     #SAI
                        78.758,     #STR
                        75.021]     #VER
})

# weather data
API_KEY = ""
City = "Barcelona"
date_of_race = "2026-06-14"                     # must follow YYYY-MM-DD
weather_url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={City}&dt={date_of_race}"
response = requests.get(weather_url)
weather_data = response.json()
tempreature = weather_data["current"]["temp_c"]


# rain probability
race_time = "16:00"
try:
    forecast_day = weather_data["forecast"]["forecastday"][0]

    chance_of_rain = None

    for hour in forecast_day["hour"]:
        if hour["time"].endswith(f"{race_time}"):
            chance_of_rain = hour["chance_of_rain"]
except Exception as e:
    print("Error", e)


# rain laptime increase
if chance_of_rain >= 25 and chance_of_rain <= 50:
    wet_factor = 1.05
elif chance_of_rain >= 50 and chance_of_rain <= 70:
    wet_factor = 1.1
else:
    wet_factor = 1.2


# chances of rain multiplier
if chance_of_rain >= 0.70:
    qualifying_2026["QualifyingTime (s)"] = qualifying_2026["QualifyingTime (s)"] * wet_factor
elif chance_of_rain >= 50 and chance_of_rain <= 70:
    qualifying_2026["QualifyingTime (s)"] = qualifying_2026["QualifyingTime (s)"] * wet_factor
else:
    qualifying_2026["QualifyingTime (s)"]

# merge fp with qualifying times
merged_data = qualifying_2026.merge(df_fp, on="Driver", how="left")


# fill na values with the recent time laps 
for col in ["FP2", "FP3", "fp_mean_best_lap"]:
    merged_data[col] = merged_data[col].fillna(merged_data["QualifyingTime (s)"])


# define features (x) and target (y)
X = merged_data[["QualifyingTime (s)", "FP2", "FP3", "Q"]]
y = merged_data[["fp_mean_best_lap"]]


# impute missing values for features
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)


# train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
model.fit(X_train, y_train)
merged_data["PredictedRacetime (s)"] = model.predict(X_imputed)


# sort results to find predicted winner
final_results = merged_data.sort_values(by=["PredictedRacetime (s)", "QualifyingTime (s)"]).reset_index(drop=True)
print(final_results[["Driver", "PredictedRacetime (s)"]])


# sort results and get top 3
podium = final_results.loc[:7, ["Driver", "PredictedRacetime (s)"]]
print("="*50)
print("🏁 Monaco Race Prediction🏁")
print("="*50)
print("\n🏆 Predicted in the top 3 🏆 \n")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
y_pred = model.predict(X_test)
print(f"Model Error (MAE) : {mean_absolute_error(y_test, y_pred):.2f} seconds")


"""
==================================================
🏁 Barcelona Race Prediction🏁
==================================================

🏆 Predicted in the top 3 🏆 

🥇 P1: RUS
🥈 P2: NOR
🥉 P3: PIA
Model Error (MAE) : 0.80 seconds
"""