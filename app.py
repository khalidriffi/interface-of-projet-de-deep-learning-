import base64
import numpy as np
import pandas as pd
import streamlit as st

# ============ SCIKIT-LEARN (ML) ============
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# Optional ML libs
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False

# ============ STATS MODELS (ARIMA / SARIMA) ============
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False

# ============ DEEP LEARNING (TF / KERAS) ============
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    Flatten,
    RepeatVector,
    TimeDistributed
)
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------
# BACKGROUND IMAGE
# -------------------------------------------------
def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image = get_base64_image("Sans titre.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{background_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.block-container {{
    backdrop-filter: blur(15px);
    background: rgba(255,255,255,0.15);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}}

h1 {{
    text-align:center;
    color:white;
    font-weight:900;
}}

h2, h3, h4, label {{
    color:white !important;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def make_sequences(series: np.ndarray, window: int):
    """
    Turn a 1D time series into (X, y) sequences.
    X shape: (n_samples, window)
    y shape: (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def evaluate_model(y_true, y_pred, name: str):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    st.write(f"### 📊 {name} Results")
    st.write("**R² Score:**", float(r2_score(y_true, y_pred)))
    st.write("**MAE:**", float(mean_absolute_error(y_true, y_pred)))


# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🤖 AI Studio — ML & Time Series Dashboard")

# -------------------------------------------------
# SIDEBAR — FILE UPLOAD & SECTION
# -------------------------------------------------
st.sidebar.header("📁 Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

section = st.sidebar.radio(
    "Choose section",
    [
        "1️⃣ Machine Learning (Tabular)",
        "2️⃣ Time Series — Statistical (ARIMA, Persistence)",
        "3️⃣ Time Series — Deep Learning"
    ]
)

if not uploaded_file:
    st.info("⬅️ Upload a CSV file to start.")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
data = pd.read_csv(uploaded_file)
st.success("Dataset loaded!")
st.dataframe(data.head())

# -------------------------------------------------
# AUTO-DETECT DATE COLUMN & SORT
# -------------------------------------------------
date_col = None
for col in data.columns:
    try:
        pd.to_datetime(data[col].dropna().iloc[0])
        date_col = col
        break
    except Exception:
        continue

if date_col:
    st.success(f"📅 Detected date column: **{date_col}**")
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col)
    data = data.set_index(date_col)
else:
    st.warning("⚠️ No date column detected. Using row index as time.")

# -------------------------------------------------
# NUMERIC COLUMNS
# -------------------------------------------------
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
if len(numeric_cols) == 0:
    st.error("❌ No numeric columns found in dataset.")
    st.stop()

# =========================================================
# 1️⃣ MACHINE LEARNING (TABULAR)
# =========================================================
if section.startswith("1️⃣"):
    st.subheader("⚙️ Machine Learning Models (Tabular Regression)")

    target_ml = st.sidebar.selectbox("Choose target column (numeric)", numeric_cols)
    feature_cols = [c for c in numeric_cols if c != target_ml]

    if len(feature_cols) == 0:
        st.error("❌ Need at least one feature column in addition to the target.")
        st.stop()

    X = data[feature_cols].copy()
    y = data[target_ml].copy()

    # Basic cleaning
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    ml_models = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "SVR",
    ]

    if XGB_AVAILABLE:
        ml_models.append("XGBoost")
    else:
        ml_models.append("XGBoost (not installed)")

    if CAT_AVAILABLE:
        ml_models.append("CatBoost")
    else:
        ml_models.append("CatBoost (not installed)")

    ml_choice = st.selectbox("Choose ML model", ml_models)

    if st.button("🚀 Train ML Model"):
        # --- Linear Regression
        if ml_choice == "Linear Regression":
            model = LinearRegression()

        elif ml_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=300, random_state=42)

        elif ml_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)

        elif ml_choice == "SVR":
            model = SVR()

        elif ml_choice.startswith("XGBoost"):
            if not XGB_AVAILABLE:
                st.error("❌ XGBoost not installed. Run: pip install xgboost")
                st.stop()
            model = XGBRegressor(random_state=42)

        elif ml_choice.startswith("CatBoost"):
            if not CAT_AVAILABLE:
                st.error("❌ CatBoost not installed. Run: pip install catboost")
                st.stop()
            model = CatBoostRegressor(verbose=0, random_state=42)

        else:
            st.error("Unknown model.")
            st.stop()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        evaluate_model(y_test, preds, ml_choice)

# =========================================================
# 2️⃣ TIME SERIES — STATISTICAL (Persistence, ARIMA, SARIMA)
# =========================================================
elif section.startswith("2️⃣"):
    st.subheader("📈 Time Series — Statistical Models")

    target_ts = st.sidebar.selectbox("Choose target (time series)", numeric_cols)

    series = data[target_ts].dropna().astype(float)

    # Chronological split (NO shuffle)
    train_ratio = st.sidebar.slider("Train ratio", 0.5, 0.95, 0.8)
    split_idx = int(len(series) * train_ratio)
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    st.write(f"Train length: {len(train_series)}, Test length: {len(test_series)}")

    ts_models = ["Persistence (Naive)"]
    if STATS_AVAILABLE:
        ts_models += ["ARIMA (1,1,1)", "SARIMA (1,1,1)x(1,1,1,7)"]
    else:
        ts_models += ["ARIMA (statsmodels not installed)", "SARIMA (statsmodels not installed)"]

    ts_choice = st.selectbox("Choose statistical model", ts_models)

    if st.button("🚀 Run Time Series Model"):
        # ---------- Persistence ----------
        if ts_choice.startswith("Persistence"):
            # y_hat_t = y_{t-1}
            # first test point uses last training value
            y_pred = pd.concat(
                [pd.Series([train_series.iloc[-1]]), test_series.iloc[:-1]]
            )
            evaluate_model(test_series, y_pred, "Persistence (Naive)")

        # ---------- ARIMA / SARIMA ----------
        else:
            if not STATS_AVAILABLE:
                st.error("❌ Install statsmodels: pip install statsmodels")
                st.stop()

            if ts_choice.startswith("ARIMA"):
                order = (1, 1, 1)
                st.write(f"Using ARIMA{order}")
                model = ARIMA(train_series, order=order)
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 7)
                st.write(f"Using SARIMA{order}x{seasonal_order}")
                model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)

            fit = model.fit()
            forecast = fit.forecast(steps=len(test_series))
            evaluate_model(test_series, forecast, ts_choice)

# =========================================================
# 3️⃣ TIME SERIES — DEEP LEARNING
# =========================================================
elif section.startswith("3️⃣"):
    st.subheader("🧠 Time Series — Deep Learning Models")

    target_dl = st.sidebar.selectbox("Choose target (time series)", numeric_cols)
    series = data[target_dl].dropna().astype(float).values.reshape(-1, 1)

    # Scaling for DL
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series).flatten()

    window = st.sidebar.slider("Window size", 5, 60, 20)
    train_ratio = st.sidebar.slider("Train ratio", 0.5, 0.95, 0.8)

    X_all, y_all = make_sequences(series_scaled, window)
    split_idx = int(len(X_all) * train_ratio)

    X_train_seq, X_test_seq = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    # Shapes
    X_train_dl = X_train_seq[..., np.newaxis]
    X_test_dl = X_test_seq[..., np.newaxis]

    # Also flattened for Dense baseline
    X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
    X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)

    dl_choice = st.selectbox(
        "Choose Deep Learning model",
        [
            "Dense NN",
            "LSTM",
            "BiLSTM",
            "GRU",
            "CNN",
            "CNN-LSTM",
            "Encoder-Decoder LSTM",
        ]
    )

    neurons = st.sidebar.slider("Hidden units", 8, 256, 64)

    if st.button("🚀 Train Deep Learning Model"):
        model = Sequential()

        # -------- Dense baseline --------
        if dl_choice == "Dense NN":
            model.add(Dense(neurons, activation="relu", input_dim=X_train_flat.shape[1]))
            model.add(Dense(neurons, activation="relu"))
            model.add(Dense(1))
            Xtr, Xte = X_train_flat, X_test_flat

        # -------- LSTM --------
        elif dl_choice == "LSTM":
            model.add(LSTM(neurons, input_shape=(window, 1)))
            model.add(Dense(1))
            Xtr, Xte = X_train_dl, X_test_dl

        # -------- BiLSTM --------
        elif dl_choice == "BiLSTM":
            model.add(Bidirectional(LSTM(neurons), input_shape=(window, 1)))
            model.add(Dense(1))
            Xtr, Xte = X_train_dl, X_test_dl

        # -------- GRU --------
        elif dl_choice == "GRU":
            model.add(GRU(neurons, input_shape=(window, 1)))
            model.add(Dense(1))
            Xtr, Xte = X_train_dl, X_test_dl

        # -------- CNN --------
        elif dl_choice == "CNN":
            model.add(Conv1D(64, 3, activation="relu", input_shape=(window, 1)))
            model.add(MaxPooling1D(2))
            model.add(Flatten())
            model.add(Dense(neurons, activation="relu"))
            model.add(Dense(1))
            Xtr, Xte = X_train_dl, X_test_dl

        # -------- CNN-LSTM --------
        elif dl_choice == "CNN-LSTM":
            model.add(Conv1D(64, 3, activation="relu", input_shape=(window, 1)))
            model.add(MaxPooling1D(2))
            model.add(LSTM(neurons))
            model.add(Dense(1))
            Xtr, Xte = X_train_dl, X_test_dl

        # -------- Encoder–Decoder LSTM --------
        elif dl_choice == "Encoder-Decoder LSTM":
            model.add(LSTM(neurons, input_shape=(window, 1)))
            model.add(RepeatVector(1))    # horizon = 1
            model.add(LSTM(neurons, return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
            Xtr, Xte = X_train_dl, X_test_dl

        else:
            st.error("Unknown DL model.")
            st.stop()

        model.compile(optimizer="adam", loss="mse")
        st.write("⏳ Training (20 epochs)...")
        model.fit(Xtr, y_train, epochs=20, batch_size=32, verbose=0)

        preds_scaled = model.predict(Xte).reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).flatten()

        # denormalize y_test too
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        evaluate_model(y_test_real, preds, dl_choice)
