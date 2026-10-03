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
Malaysia GP 2026 - Winner Prediction Model
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
    session = fastf1.get_session(2026, "Malaysia", session_name)
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

["ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "COL", "GAS", "HAM", "HUL", "LAW", "LEC", "LIN", "NOR", "OCO", "PER", "PIA", "RUS", "SAI", "STR", "TSU" ,"VER"],

"QualifyingTime (s)": [ 98.600,     #ALB
                        97.220,     #ALO
                        95.631,     #ANT
                        97.980,     #BEA
                        97.673,     #BOR
                        98.611,     #BOT
                        97.179,     #COL
                        97.210,     #GAS
                        95.558,     #HAD  
                        95.428,     #HAM
                        97.970,     #HUL
                        97.023,     #LAW
                        95.666,     #LEC
                        97.883,     #LIN
                        95.757,     #NOR
                        98.233,     #OCO
                        98.933,     #PER
                        95.762,     #PIA
                        95.871,     #RUS
                        97.527,     #SAI
                        97.566,     #STR
                        95.130]     #VER
})

# weather data
API_KEY = ""
City = "Sepang"
date_of_race = "2026-10-04"                     # must follow YYYY-MM-DD
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


# starting grid for drivers
merged_data["GridPos"] = merged_data["QualifyingTime (s)"].rank(method="first").astype(int)
grid_penalties = {
    "HAD":5,
    "LIN":5,
    "COL":5
    }
for drv, places in grid_penalties.items():
    merged_data.loc[merged_data["Driver"] == drv, "GridPos"] += places

# re-ranking the grid
merged_data["GridPos"] = merged_data["GridPos"].rank(method="first").astype(int)



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
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.15, random_state=42)


# train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=42)
model.fit(X_train, y_train)
merged_data["PredictedRacetime (s)"] = model.predict(X_imputed)


# sort results to find predicted winner
final_results = merged_data.sort_values(by=["PredictedRacetime (s)", "QualifyingTime (s)"]).reset_index(drop=True)
final_results["PredictedPos"] = final_results.index + 1
print(final_results[["Driver", "GridPos", "PredictedPos", "PredictedRacetime (s)"]])


# sort results and get top 3
print(f"\n Tempreature expected in the race is {tempreature} C")
print(f"\n Chance of rain: {chance_of_rain}%")
podium = final_results.loc[:7, ["Driver", "PredictedRacetime (s)"]]
print("="*50)
print("🏁 Malaysia Race Prediction🏁")
print("="*50)
print("\n🏆 Predicted in the top 3 🏆 \n")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
y_pred = model.predict(X_test)
print(f"Model Error (MAE) : {mean_absolute_error(y_test, y_pred):.2f} seconds")


"""
Driver  GridPos  PredictedPos  PredictedRacetime (s)
0     VER        1             1              96.757660
1     ANT        4             2              97.144241
2     LEC        8             3              97.144424
3     HAM        3             4              97.151443
4     RUS        6             5              97.175949


==================================================
🏁 Malaysia Race Prediction🏁
==================================================

🏆 Predicted in the top 3 🏆 

🥇 P1: VER
🥈 P2: ANT
🥉 P3: LEC
Model Error (MAE) : 0.12 seconds
"""