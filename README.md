# f1_2026_predictions

# 🏎️🏁 F1 Predictions 2026 - Machine Learning Model

Welcome to the **F1 Predictions 2026** repository! This project uses **machine learning, and FastF1 API data** to predict race outcomes for the 2026 Formula 1 season.

## 🚀 Project Overview
This repository contains an **XGBoost machine learning model** that predicts race results based on free practice performances, sprint results (when the weekend includes a sprint), and qualifying times. The model leverages:
- FastF1 API for free practice race data
- 2026 qualifying session results
- Feature engineering techniques to improve predictions
PS.: Over the season, I'll be adding additional features to improve our model as well

## 🔮 Do the Predictions Actually Work?
| Race       | Podium Predicted          | Actual              | Result                          |
|------------|---------------------------|---------------------|---------------------------------|
| Australia | 1.LEC/2.HAM/3.RUS       | 1.RUS/2.ANT/3.LEC  | 2/3 correct – 2 wrong position |
| China   | 1.ANT/2.RUS/3.HAM        | 1.ANT/2.RUS/3.HAM  | 3/3 podium match                |
| Japan   | 1.ANT/2.RUS/3.PIA        | 1.ANT/2.PIA/3.LEC  | 2/3 correct – 2 wrong position                |
| Miami   | 1.ANT/2.PIA/3.NOR        | 1.ANT/2.NOR/3.PIA  | 3/3 correct – 1 wrong position                |

## 📊 Data Sources
- **FastF1 API**: Fetches lap time, and telemetry data.
- **2026 Qualifying Data**: Used for prediction


## 🔍 How It Works
1. **Data Collection**: The script pulls relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering**: Converts lap times, normalizes driver names, and structures race data.
3. **Model Training**: A **XGBoosting Regressor** is trained using 2026 free practice, and sprint results.
4. **Prediction**: The model predicts race times for 2026 and ranks drivers accordingly.
5. **Evaluation**: Model performance is measured using **Mean Absolute Error (MAE)**.
6. **Weather**: Prediction of how the weather is going to be like in case it rained or not in order to add wet multiplier factor.


## ❓ FAQs
- Why am I not using historical data from the past years?
- > This year marks the first time a completely new car has been introduced due to the new regulations, making it significantly different from previous years.


### Dependencies
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `XGBoost`
- `requests`
