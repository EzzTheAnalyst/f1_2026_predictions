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
Montreal GP 2026 - Winner Prediction Model
===========================================


Approach: Train on Free Practice (FP1,SQ,S,Q)
Reasoning:
    - FP: Free Practice
    - SQ: Sprint Qualifying
    - S: Sprint
    - Qualifying: Best single-lap speed signal
    - using both gives context to the model

"""

fp_sessions = {}
for session_name in ["FP1", "SQ" ,"S", "Q"]:
    session = fastf1.get_session(2026, "Canada", session_name)
    session.load()


    # get each driver's personal best lap in the fp
    fastest = (session.laps.pick_quicklaps().groupby("Driver")["LapTime"].min().reset_index().rename(columns={"LapTime":session_name}))
    fastest[session_name] = fastest[session_name].dt.total_seconds()
    fp_sessions[session_name] = fastest


# merge all FP on driver
df_fp = fp_sessions["FP1"]
for i in ["SQ", "S", "Q"]:
    df_fp = df_fp.merge(fp_sessions[i], on="Driver", how="outer")

# mean fastest lap across all 3 sessions with ignoring na values in case a driver missed a session
df_fp["fp_mean_best_lap"] = df_fp[["FP1", "SQ" ,"S", "Q"]].mean(axis=1)

# to avoid training with NaN values
df_fp.dropna(inplace=True)


# creating a dataframe with the qualified people only and their best lap time
qualifying_2026 = pd.DataFrame({
    "Driver" : ["VER", "HAM", "LEC", "RUS", "ANT",
                "HAD", "NOR","PIA", "LAW", "LIN",
                "BOR", "HUL", "BEA", "OCO","GAS",
                "ALB", "SAI", "COL", "ALO", "PER",
                "STR", "BOT"],
"QualifyingTime (s)": [87.964, 88.319, 88.143, 88.789, 88.197,
                87.798, 88.319, 88.500, 89.499, 90.133, 
                91.967, 89.439, 89.567, 89.771, 88.81,
                89.946, 88.762, 91.098, 91.967, 91.629,
                91.164, 89.568]

})

# weather data
API_KEY = ""
City = "Montreal"
date_of_race = "2026-05-24"                     # must follow YYYY-MM-DD
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
for col in ["FP1", "SQ" ,"S", "Q", "fp_mean_best_lap"]:
    merged_data[col] = merged_data[col].fillna(merged_data["QualifyingTime (s)"])


# define features (x) and target (y)
X = merged_data[["QualifyingTime (s)", "FP1", "SQ" ,"S", "Q"]].fillna(0)
y = merged_data[["fp_mean_best_lap"]]


# impute missing values for features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=39)


# train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=3, random_state=42)
model.fit(X_train, y_train)
merged_data["PredictedRacetime (s)"] = model.predict(X_imputed)


# sort results to find predicted winner
final_results = merged_data.sort_values(by=["PredictedRacetime (s)", "QualifyingTime (s)"]).reset_index(drop=True)
print(final_results[["Driver", "PredictedRacetime (s)"]])


# sort results and get top 3
print(f"Tempreature predicted for the race is {tempreature} C \n")
podium = final_results.loc[:7, ["Driver", "PredictedRacetime (s)"]]
print("\n🏁 Shanghai Race Prediction🏁")
print("\n🏆 Predicted in the top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
y_pred = model.predict(X_test)
print(f"Model Error (MAE) : {mean_absolute_error(y_test, y_pred):.2f} seconds")


"""
   Driver  PredictedRacetime (s)
0     HUL              89.439445
1     LEC              89.451599
2     ANT              89.474953
3     PIA              89.474953
4     NOR              89.536560

🏁 Montreal Race Prediction🏁

🏆 Predicted in the top 3 🏆
🥇 P1: HUL
🥈 P2: LEC
🥉 P3: ANT
Model Error (MAE) : 0.21 seconds
"""



