# ⚡ GRU Energy Consumption Forecaster

A resume-worthy deep learning project that uses a **Gated Recurrent Unit (GRU)** neural network to forecast household energy consumption hour-by-hour.

## 🧠 What It Does

- Generates synthetic household energy data with realistic daily & weekly patterns
- Trains a 2-layer GRU model (TensorFlow/Keras) in real time inside Streamlit
- Shows live training loss curves as the model learns
- Evaluates with RMSE, MAE, MAPE metrics
- Plots: raw data, hourly usage profile, predictions vs actual, residuals, next-hour forecast

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Deep Learning | TensorFlow / Keras (GRU) |
| Data | NumPy, Pandas |
| Viz | Matplotlib |
| ML utils | Scikit-learn |
| UI | Streamlit |

---

## 🚀 Quick Start

### Requirements
- **Python 3.9 – 3.12** (TensorFlow does NOT support Python 3.13+)
- pip

### Install & Run

```bash
# 1. Clone / unzip the project
cd gru_energy_app

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🎛️ How to Use

1. Use the **sidebar** to configure:
   - Sequence length (how many past hours to feed the model)
   - GRU hidden units
   - Dropout rate
   - Epochs & batch size
   - Days of training data
2. Click **🚀 Train Model**
3. Watch the live loss curve update epoch by epoch
4. View metrics, predictions, residuals, and the next-hour forecast

---

## 📌 Resume Talking Points

- Built and trained a **GRU (Gated Recurrent Unit)** RNN for univariate time-series forecasting
- Implemented **sequence generation**, **MinMax normalisation**, and **train/val splitting** from scratch
- Evaluated model with **RMSE, MAE, MAPE**; visualised residual distribution
- Deployed an interactive ML app with **Streamlit** including live training feedback
- Handled TensorFlow/Keras model architecture design with configurable hyperparameters

---

## ⚠️ Python Version Note

TensorFlow supports **Python 3.9, 3.10, 3.11, 3.12** only.  
It does **not** support Python 3.13 or 3.14.  
Use `python --version` to verify before installing.
