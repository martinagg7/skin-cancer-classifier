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

def procesar_carpeta(input_folder, zoomed_folder, masks_folder, lesions_folder,
                     zoom_factor=0.9, size=(224, 224), nombre_csv=None):
    """
    Procesa todas las imágenes en input_folder:
       Aplica zoom
       Quita pelos
       Genera máscaras y lunares segmentados
       Calcula métricas (área, perímetro, circularidad, simetrías)
       Guarda un CSV con todas las métricas
    """

    metricas = []

    # Itera por clases Benign y Malignant
    for cls in ['Benign', 'Malignant']:
        input_path = os.path.join(input_folder, cls)
        zoom_path = os.path.join(zoomed_folder, cls)
        mask_path = os.path.join(masks_folder, cls)
        lesion_path = os.path.join(lesions_folder, cls)

        #  Verificar existencia de carpeta
        if not os.path.exists(input_path):
            print(f" Carpeta no encontrada: {input_path}")
            continue

        os.makedirs(zoom_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(lesion_path, exist_ok=True)

        print(f"Procesando conjunto: {os.path.basename(input_folder)} | Clase: {cls}")

        for img_name in tqdm(os.listdir(input_path), desc=f"Procesando {cls}"):

            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if img_name.startswith('.') or img_name.startswith('._'):
                continue  # Ignorar archivos ocultos

            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                print(f"No se pudo leer la imagen: {img_path}")
                continue

            # Zoom
            zoomed = apply_zoom(img, zoom_factor)
            zoom_output = os.path.join(zoom_path, img_name)
            cv2.imwrite(zoom_output, zoomed)

            # Quitar pelos
            rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
            clean = quitar_pelos(rgb)

            # Segmentar (devuelve máscara y lunar)
            try:
                mask, lesion_rgb = segmentar_lesion(clean, size=size)
            except Exception as e:
                print(f" Error segmentando {img_name}: {e}")
                continue

            # Guardar resultados
            mask_output = os.path.join(mask_path, img_name)
            lesion_output = os.path.join(lesion_path, img_name)

            cv2.imwrite(mask_output, mask)
            cv2.imwrite(lesion_output, cv2.cvtColor(lesion_rgb, cv2.COLOR_RGB2BGR))

            # Calcular métricas
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

    # Guardar CSV
    if nombre_csv is None:
        nombre_csv = f"metricas_{os.path.basename(input_folder)}.csv"

    df = pd.DataFrame(metricas)
    output_csv = os.path.join(masks_folder, nombre_csv)
    print(f" Guardando CSV en: {output_csv}")

    df.to_csv(output_csv, index=False)

    print(f" Total imágenes procesadas en {os.path.basename(input_folder)}: {len(metricas)}")
    print(f" Métricas guardadas en: {output_csv}")
