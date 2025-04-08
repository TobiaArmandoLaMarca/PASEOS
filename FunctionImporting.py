import ee
import geemap
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import time
import os

PROJECT_ID = "paseostest"  # Inserisci qui il tuo Project ID

ee.Initialize(project=PROJECT_ID)

# Definizione del bounding box (esempio su Roma)
bbox = ee.Geometry.Rectangle([12.3, 41.7, 12.7, 42.0])  # (min_lon, min_lat, max_lon, max_lat)

# Selezioniamo le immagini Sentinel-2 disponibili
sentinel2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
    .filterBounds(bbox) \
    .filterDate("2024-01-01", "2024-02-01") \
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
    .median()\
    .select(['B4', 'B3', 'B2'])  # Seleziona solo le bande RGB  # Media tra immagini per ridurre nuvole

# Salviamo l'immagine in formato GeoTIFF
output_path = "sentinel_image.tif"
geemap.ee_export_image(sentinel2, filename=output_path, scale=60, region=bbox)

#print(f"L'immagine è stata salvata in: {os.path.abspath(output_path)}")
# ⏳ **Aspetta che il file venga generato**
print("⏳ Attesa generazione del file...")
while not os.path.exists(output_path):
    time.sleep(2)  # Controlla ogni 2 secondi

print(f"✅ Download completato! L'immagine è stata salvata in: {os.path.abspath(output_path)}")

print("📊 Caricamento dell'immagine per la visualizzazione...")
with rasterio.open(output_path) as src:
    img = src.read([1, 2, 3])  # Carica le bande R, G, B
    img = np.moveaxis(img, 0, -1)  # Cambia ordine degli assi per matplotlib
    img = img / img.max()  # Normalizza i valori per la visualizzazione (0-1)

    # Plottiamo l'immagine
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # Rimuove assi per migliorare la visualizzazione
    #plt.title("Sentinel-2 RGB (Rosso, Verde, Blu)")
    plt.show()

