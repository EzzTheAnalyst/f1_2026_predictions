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
Belgium GP 2026 - Winner Prediction Model
===========================================


Approach: Train on Free Practice (FP1,FP2,FP3,Q)
Reasoning:
    - Qualifying: Best single-lap speed signal
    - FP: race-pace
    - Race weather (weather has a big effect on the tires and time race)
    - using both gives context to the model
"""

fp_sessions = {}
race_pace_sessions = {}                                                                                                 
for session_name in ["FP1","FP2", "FP3", "Q"]:
    session = fastf1.get_session(2026, "Belgium", session_name)
    session.load()


    # get each driver's personal best lap in the fp
    quick_laps = session.laps.pick_quicklaps()          
    fastest = (quick_laps.groupby("Driver")["LapTime"].min().reset_index().rename(columns={"LapTime":session_name}))            
    fastest[session_name] = fastest[session_name].dt.total_seconds()
    fp_sessions[session_name] = fastest

    # race pace: median lap time per driver (consistency signal)
    if session_name != "Q":  # qualifying is single-lap, not relevant for race pace
        median_col = f"{session_name}_median"
        race_pace = (quick_laps.groupby("Driver")["LapTime"].median().reset_index().rename(columns={"LapTime": median_col}))
        race_pace[median_col] = race_pace[median_col].dt.total_seconds()
        race_pace_sessions[median_col] = race_pace

# merge all FP on driver
df_fp = fp_sessions["FP1"]
for i in ["FP2","FP3", "Q"]:
    df_fp = df_fp.merge(fp_sessions[i], on="Driver", how="outer")

# merge race pace data
for col_name, race_pace_df in race_pace_sessions.items():
    df_fp = df_fp.merge(race_pace_df, on="Driver", how="outer")

# mean fastest lap across all 3 sessions with ignoring na values in case a driver missed a session
df_fp["fp_mean_best_lap"] = df_fp[["FP1","FP2","FP3","Q"]].mean(axis=1)

# mean race pace across FP sessions
df_fp["race_pace_median"] = df_fp[["FP1_median","FP2_median","FP3_median"]].mean(axis=1)

# to avoid training with NaN values
df_fp.dropna(inplace=True)


# creating a dataframe with the qualified people only and their best lap time
qualifying_2026 = pd.DataFrame({"Driver":

["ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "COL", "GAS", "HAD", "HAM", "HUL", "LAW", "LEC", "LIN", "NOR", "OCO", "PER", "PIA", "RUS", "SAI", "STR", "VER"],

"QualifyingTime (s)": [ 107.120,     #ALB
                        110.002,     #ALO
                        104.361,     #ANT
                        106.779,     #BEA
                        105.628,     #BOR
                        107.823,     #BOT
                        106.392,     #COL
                        106.331,     #GAS
                        105.823,     #HAD
                        104.895,     #HAM
                        106.671,     #HUL
                        106.120,     #LAW
                        104.893,     #LEC
                        105.143,     #LIN
                        104.801,     #NOR
                        107.801,     #OCO
                        107.971,     #PER
                        105.016,     #PIA
                        104.869,     #RUS
                        106.777,     #SAI
                        110.177,     #STR
                        104.678]     #VER
})

# weather data
API_KEY = ""
City = "Belgium"
date_of_race = "2026-07-19"                     # must follow YYYY-MM-DD
weather_url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={City}&dt={date_of_race}"
response = requests.get(weather_url)
weather_data = response.json()
tempreature = weather_data["current"]["temp_c"]


# rain probability
race_time = "15:00"
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
for col in ["FP1","FP2", "FP3", "fp_mean_best_lap"]:
    merged_data[col] = merged_data[col].fillna(merged_data["QualifyingTime (s)"])

# fill race pace NAs: use qualifying time + small offset as proxy (race pace is slower than quali)
for col in ["FP1_median","FP2_median", "FP3_median", "race_pace_median"]:
    merged_data[col] = merged_data[col].fillna(merged_data["QualifyingTime (s)"] + 1.0)

# define features (x) and target (y)
X = merged_data[["QualifyingTime (s)","FP1", "FP2", "FP3", "race_pace_median"]]
y = merged_data[["fp_mean_best_lap"]]


# impute missing values for features
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, random_state=42)


# train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=42)
model.fit(X_train, y_train)
merged_data["PredictedRacetime (s)"] = model.predict(X_imputed)


# sort results to find predicted winner
final_results = merged_data.sort_values(by=["PredictedRacetime (s)", "QualifyingTime (s)"]).reset_index(drop=True)
print(final_results[["Driver", "PredictedRacetime (s)"]])


# sort results and get top 3
print(f"\n Tempreature expected in the race is {tempreature} C")
print(f"\n Chance of rain: {chance_of_rain}%")
podium = final_results.loc[:7, ["Driver", "PredictedRacetime (s)"]]
print("="*50)
print("🏁 Belgium Race Prediction🏁")
print("="*50)
print("\n🏆 Predicted in the top 3 🏆 \n")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
y_pred = model.predict(X_test)
print(f"Model Error (MAE) : {mean_absolute_error(y_test, y_pred):.2f} seconds")


"""
==================================================
🏁 Belgium Race Prediction🏁
==================================================

🏆 Predicted in the top 3 🏆 

🥇 P1: ANT
🥈 P2: VER
🥉 P3: NOR
Model Error (MAE) : 0.26 seconds
"""