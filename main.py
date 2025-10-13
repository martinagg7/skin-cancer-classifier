from src.preprocessing.pipeline import procesar_carpeta

if __name__ == "__main__":
    # --- TRAIN ---
    procesar_carpeta(
        input_folder="data/images/train",
        zoomed_folder="data/zoomed/train",
        masks_folder="data/masks/train",
        zoom_factor=0.9,
        size=(224, 224)
    )

    # --- TEST ---
    procesar_carpeta(
        input_folder="data/images/test",
        zoomed_folder="data/zoomed/test",
        masks_folder="data/masks/test",
        zoom_factor=0.9,
        size=(224, 224)
    )

    print("✅ Zoom, máscaras y métricas generadas para TRAIN y TEST.")
