import pandas as pd
import os

def csv_to_kml(csv_path, kml_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run main.py first.")
        return

    df = pd.read_csv(csv_path)
    
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Air Pollution AQI Predictions</name>
    <description>Air quality classification mapped with coordinates. Green = Safe, Red = Polluted.</description>
    
    <!-- Style for Safe Air (Green Paddle) -->
    <Style id="safe_style">
      <IconStyle>
        <scale>1.1</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle>
        <scale>0.0</scale> <!-- Hide label by default unless hovered -->
      </LabelStyle>
    </Style>

    <!-- Style for Polluted Air (Red Paddle) -->
    <Style id="polluted_style">
      <IconStyle>
        <scale>1.1</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle>
        <scale>0.0</scale>
      </LabelStyle>
    </Style>
"""

    kml_footer = """  </Document>
</kml>
"""

    placemarks = []
    for _, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        aqi = row['Actual_AQI']
        category = row['AQI_Category']
        prob = row['Predicted_Prob']
        
        style = "#polluted_style" if category == "Polluted" else "#safe_style"
        color_indicator = "🔴" if category == "Polluted" else "🟢"
        
        placemark = f"""    <Placemark>
      <name>{category} (AQI: {int(aqi)})</name>
      <description><![CDATA[
        <h3>Air Quality Report {color_indicator}</h3>
        <p><b>Status:</b> {category}</p>
        <p><b>AQI Value:</b> {int(aqi)}</p>
        <p><b>Pollution Probability:</b> {prob:.2%}</p>
        <p><b>Coordinates:</b> {lat:.5f}, {lon:.5f}</p>
      ]]></description>
      <styleUrl>{style}</styleUrl>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>"""
        placemarks.append(placemark)

    full_kml = kml_header + "\n".join(placemarks) + kml_footer
    
    with open(kml_path, 'w', encoding='utf-8') as f:
        f.write(full_kml)
        
    print(f"KML successfully generated at: {kml_path}")

if __name__ == "__main__":
    csv_to_kml("gis/predictions.csv", "gis/predictions.kml")
