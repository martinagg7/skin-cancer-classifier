# Detección de Melanomas

Clasificación de lesiones dermatoscópicas en **benignas** o **melanoma**. El
proyecto compara tres enfoques: descriptores de forma con un modelo clásico y dos
redes convolucionales con distinta representación de entrada.

## Prueba la aplicación

Desplegada en Hugging Face: **https://huggingface.co/spaces/Martinagg/DermaScan**

Subes una foto del lunar y devuelve la probabilidad de melanoma.

---

## 1. Datos

Fotos dermatoscópicas etiquetadas como benignas (`Benign`) o melanoma
(`Malignant`).

| Conjunto | Imágenes | Origen | Uso |
|---|---|---|---|
| Entrenamiento | ~11.900 | descarga en `config/01_dowload_data.ipynb` | ajuste de los modelos |
| Test | 2.000 (1.000 por clase) | misma fuente | evaluación; también sirve de validación durante el entrenamiento |
| Externo | 45 (22 benignas / 23 malignas) | archivo público **ISIC** | prueba de generalización a otra fuente |

`data/` y `models/` no se versionan.

---

## 2. Preprocesado

Pipeline en `src/preprocessing/`, se lanza con `python main.py`.

![Pipeline de análisis](docs/img/slide_pipeline.jpg)

| Paso | Qué hace | Por qué |
|---|---|---|
| Zoom y depilado | recorte central al 90 % + eliminación de pelos (black-hat + `inpaint`) | acerca la lesión y quita el vello y el borde del dermatoscopio |
| Canal S (HSV) | se trabaja sobre el canal de saturación, suavizado con Gauss | la zona pigmentada resalta ahí con independencia de sombras y brillos |
| Segmentación | umbral de Otsu, se elige la componente conexa central y se rellenan huecos | máscara limpia de la lesión |

Salen tres representaciones de cada imagen:

| Carpeta | Contenido | La usa |
|---|---|---|
| `zoomed/` | imagen RGB recortada | ZoomNet |
| `masks/` | máscara binaria de la lesión | Random Forest |
| `lesions/` | lesión segmentada sobre fondo negro | SimpleNet |

De la máscara se extraen cinco descriptores de forma (regla ABCD), entrada del
Random Forest:

| Descriptor | Definición |
|---|---|
| Área | píxeles de la lesión |
| Perímetro | longitud del contorno |
| Circularidad | `4·π·Área / Perímetro²` (1 = círculo) |
| Simetría vertical / horizontal | solapamiento de la máscara con su reflejo respecto a cada eje |

---

## 3. Modelos

| Modelo | Entrada | Arquitectura | Accuracy test | Recall melanoma |
|---|---|---|---|---|
| Random Forest | 5 descriptores de forma | 400 árboles, prof. 10 | 0.72 | 0.65 |
| **ZoomNet** | imagen RGB con zoom | CNN, 4 bloques conv + Flatten + densa 256 | **0.89** | 0.86 |
| **SimpleNet** | lesión segmentada | CNN, 3 bloques conv + global average pooling + densa 128 | 0.81 | 0.70 |

![Esquema de la arquitectura CNN](docs/img/arquitectura_cnn.jpg)

### 3.1. Random Forest (`notebooks/00_rf_metrics.ipynb`)

Random Forest de 400 árboles sobre los cinco descriptores de forma, sin acceso a
la imagen. Es el baseline interpretable.

El análisis previo (`exploration/00_features.ipynb`) muestra que las lesiones
malignas son, en promedio, menos circulares y menos simétricas.

| Importancia de variables | Matriz de confusión (test) |
|---|---|
| <img src="docs/img/rf_feature_importance.png" width="410"> | <img src="docs/img/rf_confusion.png" width="360"> |

Accuracy 0.72. Pesan sobre todo perímetro, circularidad y área. La forma aporta
señal pero insuficiente: se le escapa el 35 % de los melanomas.

### 3.2. ZoomNet (`notebooks/01_rgb_grad_cam.ipynb`)

CNN de cuatro bloques convolucionales (32-64-128-256) sobre la imagen RGB con
zoom. Adam, `EarlyStopping` y `ReduceLROnPlateau` (para en la época 23).

![Curvas de entrenamiento de ZoomNet](docs/img/zoomnet_curvas.png)

Entrenamiento y validación van juntas, sin sobreajuste aparente. AUC 0.958.

<img src="docs/img/zoomnet_roc.png" width="360">

Mejor modelo en test interno (0.89), pero esa ventaja no se sostiene fuera
(sección 4). Los mapas Grad-CAM confirman que la activación se concentra en el
borde y el interior de la lesión:

<img src="docs/img/zoomnet_gradcam.jpg" width="760">

### 3.3. SimpleNet (`notebooks/02_simpleNet.ipynb`)

CNN más ligera: tres bloques convolucionales y *global average pooling* en lugar
de `Flatten`, lo que reduce mucho los parámetros. Entra solo la lesión
segmentada, sin contexto de piel. Adam, `EarlyStopping`.

![Curvas de entrenamiento de SimpleNet](docs/img/simplenet_curvas.png)

Accuracy 0.81 en test interno, por debajo de ZoomNet, pero con curvas muy
estables. El notebook calcula también su curva ROC.

---

## 4. Evaluación externa (ISIC)

Las dos CNN se prueban sobre 45 imágenes de ISIC, distintas en dispositivo,
iluminación y encuadre.

| Modelo | Accuracy | Recall melanoma | Recall benigno |
|---|---|---|---|
| ZoomNet | 0.58 | 0.91 (21/23) | **0.23 (5/22)** |
| SimpleNet | **0.71** | 0.83 (19/23) | 0.59 (13/22) |

| ZoomNet (externo) | SimpleNet (externo) |
|---|---|
| <img src="docs/img/zoomnet_confusion_externo.png" width="360"> | <img src="docs/img/simplenet_confusion_externo.png" width="330"> |

ZoomNet cae a 0.58 y se sesga hacia "maligno" (solo acierta 5 de 22 benignas):
al ver la imagen completa, aprende también características propias de la fuente de
entrenamiento. SimpleNet, que solo recibe la lesión segmentada, mantiene un
comportamiento equilibrado (0.71).

El conjunto es pequeño (45 imágenes), así que las cifras exactas tienen margen;
la caída y el sesgo, en cambio, son claros.

---

## 5. Modelo elegido: SimpleNet

Es el desplegado en la demo:

- Generaliza mejor a la fuente externa (0.71 vs 0.58) sin colapsar a una clase.
- Entrenamiento estable, sin sobreajuste.
- Ligero, cómodo para la demo y el dispositivo futuro.

Random Forest y ZoomNet quedan en el repositorio como parte de la comparación:
muestran que la forma sola no basta y que un buen resultado interno puede no
generalizar.

---

## 6. Estructura del repositorio

```
config/              comprobación del entorno y descarga de los datos
src/preprocessing/   zoom, depilado, segmentación y cálculo de descriptores
main.py              lanza el preprocesado sobre una carpeta de imágenes
exploration/         análisis de los descriptores de forma por clase
notebooks/
  00_rf_metrics.ipynb     Random Forest sobre los descriptores
  01_rgb_grad_cam.ipynb   ZoomNet + Grad-CAM
  02_simpleNet.ipynb      SimpleNet
reports/             caso de uso clínico y presentación
docs/img/            figuras de este README
```

---

## 7. Uso

```bash
pip install -r requirements.txt

# 1. descargar los datos: ejecutar config/01_dowload_data.ipynb

# 2. generar zoomed/, masks/, lesions/ y el CSV de descriptores
python main.py

# 3. entrenar y evaluar: ejecutar los notebooks en orden
```

---

## 8. Trabajo futuro

Dispositivo portátil basado en Raspberry Pi (cámara y pantalla) que captura la
lesión, estima la probabilidad de melanoma y envía el resultado a una plataforma
en la nube para el seguimiento clínico.

![Aplicación futura y arquitectura](docs/img/slide_futuro.jpg)
