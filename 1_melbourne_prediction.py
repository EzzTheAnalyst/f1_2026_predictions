# importing needed libraries
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


"""
Melbourne GP 2026 - Winner Prediction Model
===========================================


Approach: Train on Free Practice (FP1,FP2,FP3)
Reasoning:
    - Qualifying: Best single-lap speed signal
    - FP: race-pace
    - using both gives context to the model

"""

fp_sessions = {}
for session_name in ["FP1", "FP2", "FP3", "Q"]:
    session = fastf1.get_session(2026, "Australia", session_name)
    session.load()


    # get each driver's personal best lap in the fp
    fastest = (session.laps.pick_quicklaps().groupby("Driver")["LapTime"].min().reset_index().rename(columns={"LapTime":session_name}))
    fastest[session_name] = fastest[session_name].dt.total_seconds()
    fp_sessions[session_name] = fastest


# merge all FP on driver
df_fp = fp_sessions["FP1"]
for i in ["FP2", "FP3"]:
    df_fp = df_fp.merge(fp_sessions[i], on="Driver", how="outer")

# mean fastest lap across all 3 sessions with ignoring na values in case a driver missed a session
df_fp["fp_mean_best_lap"] = df_fp[["FP1","FP2","FP3"]].mean(axis=1)

# to avoid training with NaN values
df_fp.dropna(inplace=True)


# creating a dataframe with the qualified people only and their best lap time
qualifying_2026 = pd.DataFrame({"Driver" : ["VER", "HAM", "LEC", "HAD", "RUS",
                                 "ANT", "NOR","PIA", "LAW", "LIN",
                                 "BOR", "HUL", "BEA", "OCO","GAS",
                                 "ALB", "COL", "ALO", "PER", "BOT"],
"QualifyingTime (s)": [102.025, 79.478, 79.327, 79.303, 78.518,
    78.811, 79.475, 79.380, 79.994, 79.971, 
    80.221, 80.303, 80.311, 80.491, 80.501,
    80.941, 81.200, 81.969, 82.605, 83.244]

})


# merge fp with qualifying times
merged_data = qualifying_2026.merge(df_fp, on="Driver", how="left")


# fill na values with the recent time laps 
for col in ["FP1", "FP2", "FP3", "fp_mean_best_lap"]:
    merged_data[col] = merged_data[col].fillna(merged_data["QualifyingTime (s)"])


# define features (x) and target (y)
X = merged_data[["QualifyingTime (s)", "FP1", "FP2", "FP3"]]
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
print("\n🏁 Melbourne Race Prediction🏁")
print("\n🏆 Predicted in the top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
y_pred = model.predict(X_test)
print(f"Model Error (MAE) : {mean_absolute_error(y_test, y_pred):.2f} seconds")


"""
🏁 Melbourne Race Prediction🏁

🏆 Predicted in the top 3 🏆
🥇 P1: LEC
🥈 P2: HAM
🥉 P3: RUS
Model Error (MAE) : 0.15 seconds
"""