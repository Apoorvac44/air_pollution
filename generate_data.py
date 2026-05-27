import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2000 samples
n_samples = 2000

data = {
    'Temperature': np.random.normal(28, 6, n_samples),
    'Humidity': np.random.normal(65, 15, n_samples),
    'PM2.5': np.random.exponential(45, n_samples),
    'PM10': np.random.exponential(60, n_samples),
    # Coordinates in a realistic range (e.g., around a major city)
    'Latitude': np.random.uniform(12.8, 13.1, n_samples),
    'Longitude': np.random.uniform(77.4, 77.7, n_samples),
    'Area': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples)
}

# Calculate a realistic AQI based on PM2.5 and PM10
aqi = data['PM2.5'] * 1.5 + data['PM10'] * 0.8 + np.random.normal(0, 10, n_samples)

# Ensure AQI is bounded between 0 and 500 (standard index)
aqi = np.clip(aqi, 0, 500)
data['AQI'] = aqi

# Add some missing values to test the data cleaning step
df = pd.DataFrame(data)
missing_indices = np.random.choice(n_samples, size=50, replace=False)
df.loc[missing_indices, 'Temperature'] = np.nan
df.loc[missing_indices[:25], 'PM2.5'] = np.nan

os.makedirs('data', exist_ok=True)
df.to_csv('data/air_pollution.csv', index=False)

print("data/air_pollution.csv generated successfully!")
