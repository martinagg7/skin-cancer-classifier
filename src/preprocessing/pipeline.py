import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .zoom import apply_zoom
from .hair_removal import quitar_pelos
from .segmentation import segmentar_lesion
from .metrics import (
    calcular_area,
    calcular_perimetro,
    calcular_circularidad,
    calcular_simetria
)

def procesar_carpeta(input_folder, zoomed_folder, masks_folder, zoom_factor=0.9, size=(224, 224), nombre_csv=None):
    """
    Procesa todas las imágenes en input_folder:
      1️⃣ Aplica zoom
      2️⃣ Quita pelos
      3️⃣ Genera máscaras
      4️⃣ Calcula métricas (área, perímetro, circularidad, simetrías)
      5️⃣ Guarda un CSV con todas las métricas
    """

    os.makedirs(zoomed_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    metricas = []

    # Itera por clases Benign y Malignant
    for cls in ['Benign', 'Malignant']:
        input_path = os.path.join(input_folder, cls)
        zoom_path = os.path.join(zoomed_folder, cls)
        mask_path = os.path.join(masks_folder, cls)

        os.makedirs(zoom_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        for img_name in tqdm(os.listdir(input_path), desc=f"Procesando {cls}"):
            

            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if img_name.startswith('.') or img_name.startswith('._'):
                continue  # Ignorar archivos ocultos

            img_path = os.path.join(input_path, img_name)


            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                print(f"⚠️ No se pudo leer la imagen: {img_path}")
                continue

        
            zoomed = apply_zoom(img, zoom_factor)
            zoom_output = os.path.join(zoom_path, img_name)
            cv2.imwrite(zoom_output, zoomed)


            rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
            clean = quitar_pelos(rgb)


            try:
                mask = segmentar_lesion(clean, size=size)
            except Exception as e:
                print(f" Error segmentando {img_name}: {e}")
                continue

            mask_output = os.path.join(mask_path, img_name)
            cv2.imwrite(mask_output, mask)


            try:
                area = calcular_area(mask)
                perim = calcular_perimetro(mask)
                circ = calcular_circularidad(mask)
                sim_v, sim_h = calcular_simetria(mask)
            except Exception as e:
                print(f" Error calculando métricas en {img_name}: {e}")
                continue

            metricas.append({
                "conjunto": os.path.basename(input_folder),
                "clase": cls,
                "imagen": img_name,
                "area": area,
                "perimetro": perim,
                "circularidad": circ,
                "simetria_vertical": sim_v,
                "simetria_horizontal": sim_h
            })

    if nombre_csv is None:
        nombre_csv = f"metricas_{os.path.basename(input_folder)}.csv"

    df = pd.DataFrame(metricas)
    output_csv = os.path.join(masks_folder, nombre_csv)
    df.to_csv(output_csv, index=False)

    print(f"\n Métricas guardadas en: {output_csv}")
