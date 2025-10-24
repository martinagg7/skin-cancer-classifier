from src.preprocessing.pipeline import procesar_carpeta

if __name__ == "__main__":
    # --- EXTERNAL ---
    procesar_carpeta(
        input_folder="data/external",
        zoomed_folder="data/zoomed/external",
        masks_folder="data/masks/external",
        lesions_folder="data/lesions/external",
        zoom_factor=0.9,
        size=(224, 224)
    )

    print("✅ Zoom, máscaras y métricas generadas para EXTERNAL.")
