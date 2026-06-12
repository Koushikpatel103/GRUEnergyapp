import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRU Energy Forecaster",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    h1 { color: #00d4ff; font-family: 'Courier New', monospace; }
    h2, h3 { color: #7dd3fc; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 13px; color: #8b9ac0; margin-top: 4px; }
    .tag {
        display:inline-block; background:#1e3a5f; color:#7dd3fc;
        border-radius:5px; padding:2px 10px; font-size:12px; margin:2px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white; border: none; border-radius: 8px;
        font-weight: bold; padding: 10px 28px;
    }
    .stButton>button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n_hours: int = 1440):
    """Synthetic household energy (kWh/h) with daily & weekly patterns + noise."""
    np.random.seed(42)
    t = np.arange(n_hours)
    daily   = 0.5 * np.sin(2 * np.pi * t / 24 - np.pi / 2) + 0.5
    weekly  = 0.2 * np.sin(2 * np.pi * t / (24 * 7))
    appliance = 0.3 * np.random.rand(n_hours)          # random appliance spikes
    noise   = 0.05 * np.random.randn(n_hours)
    energy  = np.clip(daily + weekly + appliance + noise, 0, None)

    start = pd.Timestamp("2024-01-01")
    timestamps = [start + pd.Timedelta(hours=i) for i in t]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energy})
    return df


def build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def build_gru_model(seq_len: int, units: int = 64, dropout: float = 0.2):
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#1e2130")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3250")
    ax.tick_params(colors="#8b9ac0")
    ax.xaxis.label.set_color("#8b9ac0")
    ax.yaxis.label.set_color("#8b9ac0")
    return fig, ax


# ── App ───────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ GRU Energy Consumption Forecaster")
st.markdown("""
<span class='tag'>GRU (Recurrent Neural Network)</span>
<span class='tag'>Time-Series Forecasting</span>
<span class='tag'>TensorFlow / Keras</span>
<span class='tag'>Streamlit</span>
""", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "Trains a **Gated Recurrent Unit (GRU)** model on synthetic household energy data "
    "to forecast the next hour's consumption. Adjust parameters and hit **Train Model**."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Parameters")
    seq_len  = st.slider("Sequence Length (hours back)", 12, 72, 24)
    gru_units = st.selectbox("GRU Units (hidden size)", [32, 64, 128], index=1)
    dropout   = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    epochs    = st.slider("Max Epochs", 5, 50, 20)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    st.markdown("---")
    st.header("📊 Data")
    n_days = st.slider("Training Data (days)", 30, 90, 60)
    train_split = st.slider("Train / Val Split", 0.6, 0.9, 0.8, 0.05)

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True)

# ── Data preview ───────────────────────────────────────────────────────────────
df = generate_data(n_hours=n_days * 24)

st.subheader("📈 Raw Energy Data (sample)")
fig_raw, ax_raw = dark_fig(figsize=(12, 3))
ax_raw.plot(df["timestamp"][:24*14], df["energy_kwh"][:24*14],
            color="#00d4ff", lw=1.2, alpha=0.85)
ax_raw.set_xlabel("Date")
ax_raw.set_ylabel("kWh / hour")
ax_raw.set_title("First 14 Days of Household Energy", color="#7dd3fc", fontsize=11)
ax_raw.xaxis.set_major_locator(mticker.MaxNLocator(7))
plt.xticks(rotation=30)
st.pyplot(fig_raw, use_container_width=True)
plt.close()

# ── Hourly average profile ─────────────────────────────────────────────────────
st.subheader("🕐 Average Hourly Usage Profile")
df["hour"] = df["timestamp"].dt.hour
hourly_avg = df.groupby("hour")["energy_kwh"].mean()

fig_h, ax_h = dark_fig(figsize=(12, 3))
ax_h.bar(hourly_avg.index, hourly_avg.values, color="#6366f1", alpha=0.85, width=0.7)
ax_h.set_xlabel("Hour of Day")
ax_h.set_ylabel("Avg kWh")
ax_h.set_title("Average Energy Consumption by Hour", color="#7dd3fc", fontsize=11)
ax_h.set_xticks(range(0, 24))
st.pyplot(fig_h, use_container_width=True)
plt.close()

# ── Training ───────────────────────────────────────────────────────────────────
if train_btn:
    st.markdown("---")
    st.subheader("🧠 Training GRU Model")

    # Preprocess
    values = df["energy_kwh"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = build_sequences(scaled, seq_len)
    split = int(len(X) * train_split)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_gru_model(seq_len, units=gru_units, dropout=dropout)

    # Training progress
    progress_bar = st.progress(0, text="Initialising…")
    loss_history, val_loss_history = [], []

    placeholder_chart = st.empty()

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss_history.append(logs["loss"])
            val_loss_history.append(logs["val_loss"])
            pct = int((epoch + 1) / epochs * 100)
            progress_bar.progress(pct, text=f"Epoch {epoch+1}/{epochs}  |  loss: {logs['loss']:.4f}  |  val_loss: {logs['val_loss']:.4f}")

            if len(loss_history) > 1:
                fig_l, ax_l = dark_fig(figsize=(10, 3))
                ax_l.plot(loss_history, color="#00d4ff", lw=1.8, label="Train Loss")
                ax_l.plot(val_loss_history, color="#f97316", lw=1.8, linestyle="--", label="Val Loss")
                ax_l.set_xlabel("Epoch")
                ax_l.set_ylabel("MSE Loss")
                ax_l.set_title("Training Progress", color="#7dd3fc", fontsize=11)
                ax_l.legend(facecolor="#1e2130", labelcolor="#c0caf5")
                placeholder_chart.pyplot(fig_l, use_container_width=True)
                plt.close()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[StreamlitCallback(), early_stop],
        verbose=0,
    )

    progress_bar.progress(100, text="✅ Training complete!")

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_val, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{rmse:.4f}</div><div class='metric-label'>RMSE (kWh)</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{mae:.4f}</div><div class='metric-label'>MAE (kWh)</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{mape:.2f}%</div><div class='metric-label'>MAPE</div></div>", unsafe_allow_html=True)
    with c4:
        n_params = model.count_params()
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{n_params:,}</div><div class='metric-label'>Model Parameters</div></div>", unsafe_allow_html=True)

    # ── Predictions vs Actual ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Predictions vs Actual (Validation Set — last 120 hrs)")
    n_show = min(120, len(y_true))
    fig_p, ax_p = dark_fig(figsize=(12, 4))
    ax_p.plot(y_true[-n_show:], color="#00d4ff", lw=1.5, label="Actual")
    ax_p.plot(y_pred[-n_show:], color="#f97316", lw=1.5, linestyle="--", label="GRU Predicted")
    ax_p.fill_between(range(n_show), y_true[-n_show:], y_pred[-n_show:],
                      alpha=0.12, color="#f97316")
    ax_p.set_xlabel("Hours")
    ax_p.set_ylabel("kWh / hour")
    ax_p.set_title("GRU Forecast vs Ground Truth", color="#7dd3fc", fontsize=11)
    ax_p.legend(facecolor="#1e2130", labelcolor="#c0caf5")
    st.pyplot(fig_p, use_container_width=True)
    plt.close()

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.subheader("📉 Prediction Residuals")
    residuals = y_true - y_pred
    fig_r, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(12, 3.5), facecolor="#0f1117")
    for ax in (ax_r1, ax_r2):
        ax.set_facecolor("#1e2130")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2d3250")
        ax.tick_params(colors="#8b9ac0")

    ax_r1.plot(residuals[-n_show:], color="#a78bfa", lw=1.2)
    ax_r1.axhline(0, color="#f97316", lw=1, linestyle="--")
    ax_r1.set_title("Residuals over Time", color="#7dd3fc", fontsize=10)
    ax_r1.set_xlabel("Hours", color="#8b9ac0")
    ax_r1.set_ylabel("Error (kWh)", color="#8b9ac0")

    ax_r2.hist(residuals, bins=30, color="#6366f1", alpha=0.85, edgecolor="#0f1117")
    ax_r2.axvline(0, color="#00d4ff", lw=1.5, linestyle="--")
    ax_r2.set_title("Residual Distribution", color="#7dd3fc", fontsize=10)
    ax_r2.set_xlabel("Error (kWh)", color="#8b9ac0")
    ax_r2.set_ylabel("Count", color="#8b9ac0")

    plt.tight_layout()
    st.pyplot(fig_r, use_container_width=True)
    plt.close()

    # ── Next-hour forecast ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Next-Hour Forecast")
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    next_pred_scaled = model.predict(last_seq, verbose=0)
    next_kwh = scaler.inverse_transform(next_pred_scaled)[0][0]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"""
        <div class='metric-card' style='padding:30px;'>
            <div style='font-size:14px;color:#8b9ac0;'>Next Hour Prediction</div>
            <div style='font-size:48px;font-weight:bold;color:#00d4ff;margin:10px 0;'>{next_kwh:.3f}</div>
            <div style='font-size:16px;color:#7dd3fc;'>kWh</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        # Show last 24h + prediction
        last_24 = scaler.inverse_transform(scaled[-24:]).flatten()
        fig_n, ax_n = dark_fig(figsize=(8, 3.5))
        ax_n.plot(range(24), last_24, color="#00d4ff", lw=1.8, label="Last 24 hrs")
        ax_n.plot(24, next_kwh, "o", color="#f97316", markersize=12, zorder=5, label="Predicted next hr")
        ax_n.plot([23, 24], [last_24[-1], next_kwh], "--", color="#f97316", lw=1.5)
        ax_n.set_xlabel("Hour offset")
        ax_n.set_ylabel("kWh")
        ax_n.set_title("Last 24 hrs + Next-Hour Prediction", color="#7dd3fc", fontsize=10)
        ax_n.legend(facecolor="#1e2130", labelcolor="#c0caf5")
        st.pyplot(fig_n, use_container_width=True)
        plt.close()

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏗️ GRU Architecture Summary")
    arch_data = []
    for layer in model.layers:
        try:
            # Works in TF 2.x when model has been built/called
            out_shape = str(layer.output.shape)
        except Exception:
            out_shape = "—"
        arch_data.append({
            "Layer": layer.name,
            "Type": type(layer).__name__,
            "Output Shape": out_shape,
            "Parameters": f"{layer.count_params():,}",
        })
    st.dataframe(
        pd.DataFrame(arch_data),
        use_container_width=True,
        hide_index=True,
    )

    st.success("✅ Model trained successfully! Check the metrics and charts above.")

else:
    st.info("👈 Configure parameters in the sidebar and click **Train Model** to start.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b9ac0;font-size:12px;'>"
    "GRU Energy Forecaster · Built with TensorFlow/Keras + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
