# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

st.title("Rossmann Store Sales Forecast")

# -----------------------------
# Model paths
# -----------------------------
MODEL_PATH_TXT = Path("models/gbm_model.txt")
MODEL_PATH_JOBLIB = Path("models/final_model.joblib")

model = None
encoder = None

if MODEL_PATH_TXT.exists():
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(MODEL_PATH_TXT))
        st.success("✅ Loaded LightGBM booster model")
    except Exception:
        st.error("LightGBM model file found but LightGBM package not available. ➡ Please install it by running: pip install lightgbm")
elif MODEL_PATH_JOBLIB.exists():
    model = joblib.load(MODEL_PATH_JOBLIB)
    st.success("✅ Loaded joblib model")
else:
    st.error("❌ No model found in models/. Place models/gbm_model.txt or models/final_model.joblib")

# -----------------------------
# Data
# -----------------------------
DATA_PATH = Path("data/train.csv")
train = pd.DataFrame()
if DATA_PATH.exists():
    train = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    st.info("Historical train.csv loaded.")
else:
    st.warning("Place the historical train.csv at data/train.csv to enable forecasting.")

# -----------------------------
# Feature functions
# -----------------------------
def prepare_calendar_features(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    return df

def create_lag_features(df, lags=[1, 7, 14], windows=[7, 14]):
    df = df.sort_values(["Store", "Date"]).copy()
    for lag in lags:
        df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
    for w in windows:
        df[f"Sales_roll_mean_{w}"] = (
            df.groupby("Store")["Sales"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df

# -----------------------------
# Inputs
# -----------------------------
store_id = st.number_input("Store ID", min_value=1, value=1)
horizon = st.slider("Forecast horizon (days)", 1, 30, 14)

# -----------------------------
# Forecasting
# -----------------------------
if st.button("Forecast"):
    if model is None or train.empty:
        st.error("Model or historical data missing")
    else:
        # Prepare training data features
        train = prepare_calendar_features(train)

        # Encode categorical features
        for col in ["StoreType", "Assortment"]:
            if col in train.columns:
                if encoder is None:
                    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    encoder.fit(train[[col]].fillna("NA"))
                train[col + "_enc"] = encoder.transform(train[[col]].fillna("NA"))

        # Take store history
        store_hist = train[train["Store"] == store_id].copy()
        store_hist = create_lag_features(store_hist)
        last_date = store_hist["Date"].max()
        future_df = store_hist.copy()

        future_preds = []

        for d in range(1, horizon + 1):
            new_date = last_date + pd.Timedelta(days=d)
            row = {"Store": store_id, "Date": new_date}

            # Carry forward categorical info
            for col in ["StoreType", "Assortment"]:
                if col in store_hist.columns:
                    row[col] = store_hist[col].iloc[-1]

            temp = pd.DataFrame([row])
            temp = prepare_calendar_features(temp)

            # Encode categorical cols
            for col in ["StoreType", "Assortment"]:
                if col in temp.columns and encoder is not None:
                    temp[col + "_enc"] = encoder.transform(temp[[col]].fillna("NA"))

            # Merge into history to create lag features
            temp_full = pd.concat([future_df, temp], ignore_index=True)
            temp_full = create_lag_features(temp_full)

            # Select last row
            temp_ready = temp_full.iloc[[-1]]

            # Align features with model
            feat_names = model.feature_name()
            missing_feats = [f for f in feat_names if f not in temp_ready.columns]
            for mf in missing_feats:
                temp_ready[mf] = 0

            X_future = temp_ready[feat_names].fillna(0)

            # Predict
            pred = model.predict(X_future)[0]
            pred = round(float(pred), 2)  # round to 2 dp
            future_preds.append({"Date": new_date, "PredictedSales": f"${pred:,.2f}"})

            # Append prediction to history for lag continuity
            temp_full.loc[temp_full.index[-1], "Sales"] = pred
            future_df = temp_full.copy()

        forecast_df = pd.DataFrame(future_preds)
        st.subheader(f"Forecast for Store {store_id}")
        st.dataframe(forecast_df)

        # Plot (numeric version for chart)
        forecast_chart = forecast_df.copy()
        forecast_chart["PredictedSales"] = forecast_chart["PredictedSales"].replace("[\$,]", "", regex=True).astype(float)

        st.line_chart(forecast_chart.set_index("Date"))
