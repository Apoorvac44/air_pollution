# ============================================================
# STREAMLIT INTERACTIVE DASHBOARD
# Air Pollution Prediction & Analytics
# Run: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Air Pollution AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #1a73e8;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .kpi-value { font-size: 2rem; font-weight: 800; }
    .kpi-label { font-size: 0.85rem; opacity: 0.9; margin-top: 4px; }
    .safe-badge {
        background: #28a745; color: white; padding: 6px 16px;
        border-radius: 20px; font-weight: 600; font-size: 1.1rem;
    }
    .polluted-badge {
        background: #dc3545; color: white; padding: 6px 16px;
        border-radius: 20px; font-weight: 600; font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🌫️ Air Pollution AQI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">1D CNN Deep Learning Model · GIS Integration · Real-time Simulation</div>', unsafe_allow_html=True)
st.divider()


# ── Load artefacts ───────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.models import load_model as keras_load
        return keras_load('model/cnn_aqi_model.h5')
    except Exception as e:
        st.warning(f"Model not found – run main.py first. ({e})")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('dashboard/full_predictions.csv')
    except Exception:
        return None

model  = load_model()
scaler = load_scaler()
df     = load_data()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Navigation")
    page = st.radio(
        "Select Section",
        ["📊 KPI Overview", "🔮 Live Prediction", "🗺️ GIS Map", "📈 Analytics"]
    )
    st.divider()
    if df is not None:
        st.metric("Dataset Rows", f"{len(df):,}")
        if 'Predicted_Class' in df.columns:
            polluted_pct = round(df['Predicted_Class'].mean() * 100, 1)
            st.metric("Pollution Rate", f"{polluted_pct} %")


# ============================================================
# PAGE 1 — KPI OVERVIEW
# ============================================================
if page == "📊 KPI Overview":
    st.subheader("📊 Key Performance Indicators")

    if os.path.exists('dashboard/kpi_summary.csv'):
        kpi = pd.read_csv('dashboard/kpi_summary.csv').iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🧾 Total Samples",      f"{int(kpi['Total_Samples']):,}")
        with c2:
            st.metric("✅ Safe",               f"{int(kpi['Safe_Count']):,}",
                      f"{kpi['Safe_Percent']} %")
        with c3:
            st.metric("⚠️ Polluted",           f"{int(kpi['Polluted_Count']):,}",
                      f"{kpi['Polluted_Percent']} %", delta_color='inverse')
        with c4:
            st.metric("🎯 Model Accuracy",     f"{kpi['Model_Accuracy_Pct']} %")

        st.divider()
        c5, c6, c7 = st.columns(3)
        with c5:  st.metric("📉 Min AQI", kpi['Min_AQI'])
        with c6:  st.metric("📊 Mean AQI", kpi['Mean_AQI'])
        with c7:  st.metric("📈 Max AQI", kpi['Max_AQI'])
    else:
        st.info("Run **main.py** first to generate KPI data.")

    # AQI distribution chart
    st.divider()
    st.subheader("AQI Distribution")
    if df is not None and 'AQI' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df['AQI'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(100, color='red', linestyle='--', linewidth=2, label='Threshold (100)')
        ax.set_xlabel('AQI'); ax.set_ylabel('Count')
        ax.set_title('AQI Distribution'); ax.legend()
        st.pyplot(fig)
        plt.close()


# ============================================================
# PAGE 2 — LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.subheader("🔮 Real-Time AQI Prediction")
    st.markdown("Enter sensor readings to get an instant AQI classification.")

    c1, c2 = st.columns(2)
    with c1:
        temp     = st.slider("🌡️ Temperature (°C)",   5.0,  50.0, 28.0, 0.5)
        humidity = st.slider("💧 Humidity (%)",        10.0, 100.0, 65.0, 1.0)
    with c2:
        pm25 = st.slider("🔬 PM2.5 (µg/m³)",          0.0,  300.0, 45.0, 1.0)
        pm10 = st.slider("🏭 PM10 (µg/m³)",            0.0,  500.0, 80.0, 1.0)

    if st.button("🚀 Predict AQI Class", use_container_width=True):
        if model and scaler:
            inp     = np.array([[temp, humidity, pm25, pm10]])
            scaled  = scaler.transform(inp)
            cnn_inp = scaled.reshape(1, 4, 1)
            prob    = float(model.predict(cnn_inp, verbose=0)[0][0])
            cls     = 1 if prob > 0.5 else 0

            st.divider()
            rc1, rc2 = st.columns(2)
            with rc1:
                if cls == 1:
                    st.markdown('<div class="polluted-badge">⚠️ POLLUTED AIR</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-badge">✅ SAFE AIR</div>',
                                unsafe_allow_html=True)
            with rc2:
                st.progress(prob, text=f"Pollution probability: {prob:.2%}")
        else:
            st.error("Model not loaded. Run **main.py** first.")

    # ── Real-time simulation ──────────────────────────────────
    st.divider()
    st.subheader("📡 Real-Time AQI Simulation")
    st.markdown("Simulates AQI readings over time using random sensor noise.")

    if st.button("▶️ Run Simulation (50 ticks)"):
        times, aqis, classes = [], [], []
        prog = st.progress(0)
        chart_placeholder = st.empty()

        base_pm25, base_pm10 = pm25, pm10
        for i in range(50):
            t_val  = temp     + np.random.normal(0, 1)
            h_val  = humidity + np.random.normal(0, 2)
            p25    = max(0, base_pm25 + np.random.normal(0, 10))
            p10    = max(0, base_pm10 + np.random.normal(0, 15))
            aqi_est = p25 * 1.5 + p10 * 0.8 + np.random.normal(0, 5)

            times.append(i + 1)
            aqis.append(round(aqi_est, 1))
            classes.append('Polluted' if aqi_est >= 100 else 'Safe')
            prog.progress((i + 1) / 50)

            if (i + 1) % 10 == 0:
                sim_df = pd.DataFrame({'Tick': times, 'AQI': aqis, 'Class': classes})
                fig, ax = plt.subplots(figsize=(10, 3))
                colors = ['red' if c == 'Polluted' else 'green' for c in classes]
                ax.bar(times, aqis, color=colors, alpha=0.7)
                ax.axhline(100, color='red', linestyle='--', linewidth=1.5)
                ax.set_xlabel('Tick'); ax.set_ylabel('AQI')
                ax.set_title('Live AQI Simulation')
                chart_placeholder.pyplot(fig)
                plt.close()

        st.success("Simulation complete!")


# ============================================================
# PAGE 3 — GIS MAP
# ============================================================
elif page == "🗺️ GIS Map":
    st.subheader("🗺️ Geospatial Pollution Map")

    if os.path.exists('gis/predictions.csv'):
        gis = pd.read_csv('gis/predictions.csv')
        gis['color'] = gis['Predicted_Class'].map({0: '#28a745', 1: '#dc3545'})

        st.markdown("**Green = Safe   |   Red = Polluted**")
        # Streamlit's built-in scatter map
        st.map(gis.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))

        st.divider()
        st.subheader("Pollution Heatmap (Prediction Probability)")
        # Simple heatmap grid
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            gis['Longitude'], gis['Latitude'],
            c=gis['Predicted_Prob'], cmap='RdYlGn_r',
            s=18, alpha=0.6
        )
        plt.colorbar(sc, ax=ax, label='Pollution Probability')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('AQI Pollution Heatmap', fontweight='bold')
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("📥 Download GIS Data")
        st.download_button(
            "⬇️ Download predictions.csv (QGIS-ready)",
            data=gis.to_csv(index=False),
            file_name='predictions.csv',
            mime='text/csv'
        )
    else:
        st.info("Run **main.py** first to generate GIS predictions.")


# ============================================================
# PAGE 4 — ANALYTICS
# ============================================================
elif page == "📈 Analytics":
    st.subheader("📈 Model Analytics")

    # Training history charts (saved PNGs)
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('outputs/training_history.png'):
            st.image('outputs/training_history.png',
                     caption='Training Accuracy & Loss', use_column_width=True)
    with col2:
        if os.path.exists('outputs/confusion_matrix.png'):
            st.image('outputs/confusion_matrix.png',
                     caption='Confusion Matrix', use_column_width=True)

    st.divider()

    # Area-wise breakdown
    if os.path.exists('dashboard/area_summary.csv'):
        st.subheader("🗂️ Area-wise Pollution Breakdown")
        area_df = pd.read_csv('dashboard/area_summary.csv')
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(area_df.style.background_gradient(
                subset=['Pollution_Rate_Pct'], cmap='RdYlGn_r'))
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(area_df['Area'], area_df['Pollution_Rate_Pct'],
                   color=plt.cm.RdYlGn_r(area_df['Pollution_Rate_Pct'] / 100))
            ax.set_xlabel('Area'); ax.set_ylabel('Pollution Rate (%)')
            ax.set_title('Pollution Rate by Area')
            st.pyplot(fig); plt.close()

    st.divider()

    # Raw predictions table with filtering
    if df is not None:
        st.subheader("🔍 Explore Predictions")
        cat_filter = st.multiselect("Filter by AQI Category",
                                    ['Safe', 'Polluted'], default=['Safe', 'Polluted'])
        if 'AQI_Category' in df.columns:
            filtered = df[df['AQI_Category'].isin(cat_filter)]
        else:
            filtered = df
        st.dataframe(filtered.head(200), use_container_width=True)
        st.download_button("⬇️ Download Full Predictions",
                           data=df.to_csv(index=False),
                           file_name='full_predictions.csv',
                           mime='text/csv')
