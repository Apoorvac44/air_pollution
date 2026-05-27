# 🌫️ Air Pollution Prediction & GIS Dashboard

> **A complete end-to-end Deep Learning project** — 1D CNN · AQI Classification · GIS Integration · Power BI Exports · Streamlit Dashboard

---

## 📌 Project Overview

This project predicts air pollution levels (AQI) using a **1D Convolutional Neural Network (CNN)** built with TensorFlow/Keras. It includes:

- ✅ Full ML pipeline (cleaning → training → evaluation)
- ✅ GIS-ready CSV output (QGIS-compatible)
- ✅ Power BI–ready dashboard exports
- ✅ Interactive Streamlit dashboard with real-time prediction
- ✅ Heatmap + simulation features

---

## 📁 Folder Structure

```
AirPollutionProject/
├── data/
│   └── air_pollution.csv          ← Input dataset
├── model/
│   ├── cnn_aqi_model.h5           ← Trained CNN model
│   └── scaler.pkl                 ← StandardScaler (for inference)
├── outputs/
│   ├── aqi_distribution.png       ← AQI histogram
│   ├── training_history.png       ← Accuracy & Loss curves
│   └── confusion_matrix.png       ← Confusion matrix heatmap
├── dashboard/
│   ├── full_predictions.csv       ← All samples + predictions
│   ├── kpi_summary.csv            ← KPI metrics for Power BI cards
│   └── area_summary.csv           ← Area-wise pollution breakdown
├── gis/
│   └── predictions.csv            ← QGIS-ready lat/lon + predictions
├── generate_data.py               ← Synthetic dataset generator
├── main.py                        ← End-to-end ML pipeline
├── streamlit_app.py               ← Interactive dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_data.py
```

### 3. Run the Full Pipeline
```bash
python main.py
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

---

## 🧠 Model Architecture

```
Input (4 features) → reshape to (4, 1)
  ↓
Conv1D (64 filters, kernel=2, ReLU)
MaxPooling1D
Dropout (0.25)
  ↓
Conv1D (128 filters, kernel=2, ReLU)
MaxPooling1D
Dropout (0.25)
  ↓
Flatten
Dense (128, ReLU)
Dropout (0.30)
Dense (64, ReLU)
  ↓
Dense (1, Sigmoid) → [0=Safe, 1=Polluted]
```

- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Callbacks**: EarlyStopping + ModelCheckpoint

---

## 📊 AQI Classification Logic

| AQI Value | Class | Label |
|-----------|-------|-------|
| < 100     | 0     | ✅ Safe |
| ≥ 100     | 1     | ⚠️ Polluted |

---

## 📈 Evaluation Metrics

The pipeline outputs:
- **Accuracy**
- **Precision / Recall / F1-Score** (per class)
- **Confusion Matrix**

---

## 🗺️ GIS Integration (QGIS)

1. Open **QGIS**
2. `Layer → Add Layer → Add Delimited Text Layer`
3. Browse to `gis/predictions.csv`
4. Set `Latitude` and `Longitude` as geometry columns
5. Style by `AQI_Category` field (Green = Safe, Red = Polluted)

---

## 📊 Power BI Integration

Import these files into Power BI:

| File | Purpose |
|------|---------|
| `dashboard/full_predictions.csv` | Main data table |
| `dashboard/kpi_summary.csv` | KPI card values |
| `dashboard/area_summary.csv` | Bar/pie chart source |
| `gis/predictions.csv` | Map visual (lat/lon) |

**Recommended visuals:**
- Card → Model Accuracy, Total Samples
- Pie/Donut → Safe vs Polluted counts
- Map → Plot lat/lon colored by AQI_Category
- Line Chart → AQI trend (by area or index)

---

## 🎛️ Streamlit Dashboard Features

| Page | Features |
|------|----------|
| 📊 KPI Overview | Accuracy, safe/polluted counts, AQI stats |
| 🔮 Live Prediction | Slider inputs → instant CNN prediction |
| 📡 Real-time Simulation | 50-tick AQI simulation with live chart |
| 🗺️ GIS Map | Interactive scatter map + heatmap |
| 📈 Analytics | Training charts, area breakdown, data table |

---

## 🔧 Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| Temperature | Ambient temperature | °C |
| Humidity | Relative humidity | % |
| PM2.5 | Fine particulate matter | µg/m³ |
| PM10 | Coarse particulate matter | µg/m³ |

---

## 💼 Use Cases

- 🎓 Academic mini-project / final year project
- 💼 Internship portfolio piece
- 📄 Resume project (ML + GIS + Dashboard)
- 🏛️ Academic presentation / demo

---

## 📦 Requirements

```
numpy, pandas, matplotlib, seaborn
scikit-learn, tensorflow
streamlit
```

---

## 👤 Author

**Air Quality Research Team**  
Built with Python 3.10+ · TensorFlow 2.x · Streamlit
