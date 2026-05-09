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
Miami GP 2026 - Winner Prediction Model
===========================================


Approach: Train on Free Practice (FP1,SQ,S,Q)
Reasoning:
    - Qualifying: Best single-lap speed signal
    - FP: race-pace
    - SQ: Sprint Qualifying
    - S: Sprint
    - using both gives context to the model

"""

fp_sessions = {}
for session_name in ["FP1", "SQ" ,"S", "Q"]:
    session = fastf1.get_session(2026, "Miami", session_name)
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

# get weather data for miami
weather_url = ("https://api.open-meteo.com/v1/forecast"
    "?latitude=25.7743&longitude=-80.1937"
    "&hourly=precipitation_probability,precipitation,temperature_2m"
    "&timezone=auto"
    "&start_date=2026-05-03"
    "&end_date=2026-05-03"
)
response = requests.get(weather_url)
weather_data = response.json()
target_time = "2026-05-03T16:00"
hourly_times = weather_data["hourly"]["time"]


if target_time in hourly_times:
    index = hourly_times.index(target_time)
    rain_probability = weather_data["hourly"]["precipitation"][index]
    temperature = weather_data["hourly"]["temperature_2m"][index]
else:
    rain_probability = 0
    temperature = 0

rain_prob = rain_probability / 100


if rain_prob >= 0.75:
    wet_multiplyer = 1 + (rain_prob * 0.1)
    qualifying_2026["QualifyingTime"] = qualifying_2026["QualifyingTime (s)"] * wet_multiplyer
else:
    qualifying_2026["QualifyingTime"] = qualifying_2026["QualifyingTime (s)"]

df_fp["Rain Probability"] = rain_probability
df_fp["Tempreature"] = temperature

# merge fp with qualifying times
merged_data = qualifying_2026.merge(df_fp, on="Driver", how="left")




# fill na values with the recent time laps 
for col in ["FP1", "SQ" ,"S", "Q", "fp_mean_best_lap"]:
    if col in qualifying_2026.columns:
        qualifying_2026[f"{col}_wet"] = qualifying_2026[col] * wet_multiplyer
    merged_data[col] = merged_data[col].fillna(merged_data["QualifyingTime (s)"])


# define features (x) and target (y)
X = merged_data[["QualifyingTime (s)", "FP1", "SQ" ,"S", "Q", "QualifyingTime"]].fillna(0)
y = merged_data[["fp_mean_best_lap"]]


# impute missing values for features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=39)


# train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=42)
model.fit(X_train, y_train)
merged_data["PredictedRacetime (s)"] = model.predict(X_imputed)


# sort results to find predicted winner
final_results = merged_data.sort_values(by=["PredictedRacetime (s)", "QualifyingTime (s)"]).reset_index(drop=True)
print(final_results[["Driver", "PredictedRacetime (s)"]])


# sort results and get top 3
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
0     ANT              92.913116
1     RUS              92.913116
2     HAM              93.407532
3     LEC              93.407532
4     PIA              93.407532

🏁 Shanghai Race Prediction🏁

🏆 Predicted in the top 3 🏆
🥇 P1: ANT
🥈 P2: RUS
🥉 P3: HAM
Model Error (MAE) : 0.08 seconds
"""

print(rain_prob)
print(rain_probability)