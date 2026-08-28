# Detección de Melanomas

El melanoma es el cáncer de piel más agresivo y difícil de detectar a simple
vista. Los dermatólogos usan la regla **ABCD** (Asimetría, Bordes, Color,
Diámetro), pero aun así hay melanomas que pasan desapercibidos.

Este proyecto clasifica lesiones de la piel en **benignas** o **melanoma** a
partir de imágenes dermatoscópicas.

## Prueba la aplicación

Está desplegada en Hugging Face: **https://huggingface.co/spaces/Martinagg/DermaScan**

Subes una foto de la lesión y devuelve la probabilidad de melanoma.

---

## 1. Objetivo

Comparar dos maneras de decidir si un lunar es benigno o melanoma:

1. **Usando solo la forma del lunar.** Se miden características geométricas (área,
   perímetro, lo redondo que es, si es simétrico) y un modelo sencillo decide con
   esos números. Ventaja: se entiende por qué toma cada decisión.
2. **Usando la imagen completa.** Una red neuronal aprende directamente de las
   fotos, sin que nadie le diga qué mirar. Se prueban dos redes y se comprueba
   cuál acierta mejor con imágenes que no ha visto nunca.

---

## 2. Datos

Imágenes dermatoscópicas etiquetadas en dos clases (`Benign`, `Malignant`).

| Conjunto | Imágenes | Uso |
|---|---|---|
| Entrenamiento | ~11.900 | ajuste de los modelos |
| Test interno | 2.000 (1.000 por clase) | evaluación y validación durante el entrenamiento |
| Externo (ISIC) | 45 (22 benignas / 23 malignas) | prueba de generalización a otra fuente |

El dataset se descarga con `config/01_dowload_data.ipynb`. Las carpetas `data/` y
`models/` no se versionan.

---

## 3. Pipeline de preprocesado

Todo el preprocesado está en `src/preprocessing/` y se lanza con `python main.py`.

![Pipeline de análisis](docs/img/slide_pipeline.jpg)

| Paso | Qué hace | Por qué |
|---|---|---|
| **Zoom + limpieza** | recorte central al 90 % y eliminación de pelos (black-hat + `inpaint`) | centra la lesión y elimina bordes del dermatoscopio y vello que la ocultan |
| **Espacio HSV** | se toma el canal de saturación (S) y se suaviza | resalta la zona pigmentada con independencia del ruido de la imagen |
| **Máscara final** | umbral de Otsu, selección de la componente conexa central y relleno de huecos | separa la lesión del fondo |

De aquí salen tres representaciones de cada imagen, cada una alimenta a un modelo
distinto:

| Salida | Contenido | La usa |
|---|---|---|
| `zoomed/` | imagen RGB recortada | ZoomNet |
| `masks/` | máscara binaria del lunar | Random Forest (vía métricas) |
| `lesions/` | lesión segmentada sobre fondo negro | SimpleNet |

De la máscara se calculan cinco medidas de forma: área, perímetro, circularidad
(`4·π·A / P²`) y simetría vertical y horizontal. Se guardan en un CSV.

---

## 4. Modelos planteados

| Modelo | Entrada | Idea | Accuracy test | Recall melanoma |
|---|---|---|---|---|
| Random Forest | 5 métricas de forma | baseline interpretable (regla ABCD) | 0.72 | 0.65 |
| **ZoomNet** | imagen RGB con zoom | CNN que ve la lesión y su entorno | **0.89** | 0.86 |
| **SimpleNet** | lesión segmentada | CNN ligera, solo la lesión | 0.81 | 0.70 |

Esquema general de las dos CNN: bloques de convolución + pooling para extraer
características y capas densas + softmax para clasificar. ZoomNet y SimpleNet
varían el número de bloques y la capa final, como se detalla abajo.

![Arquitectura de la red CNN](docs/img/arquitectura_cnn.jpg)

### 4.1. Random Forest (`notebooks/00_rf_metrics.ipynb`)

Modelo clásico entrenado **solo con los cinco descriptores morfológicos**, sin ver
la imagen. Sirve para medir cuánta información diagnóstica hay en la pura forma del
lunar.

- `RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)`.
- Antes se hace un análisis exploratorio de cada descriptor por clase
  (`exploration/00_features.ipynb`): las lesiones malignas tienden a ser menos
  circulares y menos simétricas.

| Importancia de variables | Matriz de confusión (test) |
|---|---|
| <img src="docs/img/rf_feature_importance.png" width="420"> | <img src="docs/img/rf_confusion.png" width="360"> |

Las variables más influyentes son **perímetro, circularidad y área**. El modelo
llega a 0.72 de accuracy: la forma aporta señal, pero se queda corta (deja pasar
el 35 % de los melanomas).

### 4.2. ZoomNet: CNN sobre la imagen RGB (`notebooks/01_rgb_grad_cam.ipynb`)

CNN entrenada desde cero sobre la imagen **RGB con zoom** (224×224×3). Ve la lesión
**y la piel de alrededor**.

```
Entrada 224x224x3
 → 4 x [Conv2D(3x3, ReLU, padding same) + MaxPooling2D(2x2)]   filtros 32, 64, 128, 256
 → Dropout(0.2)
 → Flatten → Dense(256, ReLU, L2=1e-3) → Dropout(0.5)
 → Dense(2, softmax)
```

Optimizador Adam (lr 1e-3), `categorical_crossentropy`, con `EarlyStopping` y
`ReduceLROnPlateau` (se detiene sobre la época 23).

**Curvas de entrenamiento y validación** (salida directa del notebook):

![Curvas de entrenamiento de ZoomNet](docs/img/zoomnet_curvas.png)

**Curva ROC (test):**

<img src="docs/img/zoomnet_roc.png" width="360">

Es el **mejor modelo en test interno**: accuracy 0.89 y AUC 0.958, con curvas de
entrenamiento y validación que van juntas. Esa ventaja, sin embargo, no se
mantiene con imágenes de otra fuente (sección 5).

**Grad-CAM.** Para comprobar en qué se fija la red se generan mapas de calor sobre
la última capa convolucional. La red se centra en el borde y el interior de la
lesión, no en el fondo:

<img src="docs/img/zoomnet_gradcam.jpg" width="760">

### 4.3. SimpleNet: CNN sobre la lesión segmentada (`notebooks/02_simpleNet.ipynb`)

CNN **ligera** entrenada sobre la **lesión ya segmentada** (fondo negro), para que
no pueda aprender ruido del entorno.

```
Entrada 224x224x3
 → Conv2D(32) + MaxPooling2D
 → Conv2D(64) + MaxPooling2D
 → Conv2D(128) → GlobalAveragePooling2D
 → Dense(128, ReLU, L2=1e-4) → Dropout(0.4)
 → Dense(2, softmax)
```

`GlobalAveragePooling` en lugar de `Flatten` reduce mucho los parámetros. Adam
(lr 1e-4), `EarlyStopping(patience=10)`, 40 épocas máximas.

**Curvas de entrenamiento y validación** (salida directa del notebook):

![Curvas de entrenamiento de SimpleNet](docs/img/simplenet_curvas.png)

Accuracy 0.81 en test interno, por debajo de ZoomNet, pero con curvas **muy
estables y paralelas**, sin sobreajuste. El notebook incluye también su curva ROC
y su AUC (celda "Curva ROC").

---

## 5. Evaluación en datos externos (ISIC)

Prueba sobre 45 imágenes de ISIC, una fuente distinta a la de entrenamiento.

| Modelo | Accuracy externo | Recall melanoma | Recall benigno |
|---|---|---|---|
| ZoomNet | 0.58 | 0.91 (21/23) | **0.23 (5/22)** |
| SimpleNet | **0.71** | 0.83 (19/23) | 0.59 (13/22) |

| ZoomNet (externo) | SimpleNet (externo) |
|---|---|
| <img src="docs/img/zoomnet_confusion_externo.png" width="360"> | <img src="docs/img/simplenet_confusion_externo.png" width="330"> |

ZoomNet, el mejor en test interno, cae a 0.58 y se sesga hacia "maligno" (solo
acierta 5 de 22 benignas): ve la imagen completa y aprende pistas propias de la
fuente de entrenamiento. SimpleNet, que solo ve la lesión segmentada, se mantiene
equilibrado (0.71) con datos nuevos.

---

## 6. Modelo elegido: SimpleNet

Se despliega **SimpleNet** en la demo [DermaScan](https://huggingface.co/spaces/Martinagg/DermaScan):

- **Generaliza mejor.** Sube de 0.58 (ZoomNet) a 0.71 de accuracy en el conjunto
  externo, y sin colapsar hacia una sola clase.
- **Entrenamiento estable.** Curvas de entrenamiento y validación paralelas, sin
  sobreajuste.
- **Ligera.** Menos parámetros gracias al `GlobalAveragePooling`, lo que facilita
  la inferencia en la demo y en el dispositivo futuro.

ZoomNet y el Random Forest se conservan en el repositorio como parte de la
comparación: muestran, respectivamente, que las métricas internas pueden engañar
y que la forma por sí sola no basta.

---

## 7. Estructura del repositorio

```
config/              comprobación del entorno y descarga del dataset
src/preprocessing/   zoom, limpieza, segmentación y cálculo de métricas
main.py              ejecuta el pipeline sobre una carpeta de imágenes
exploration/         análisis exploratorio de los descriptores de forma
notebooks/
  00_rf_metrics.ipynb     Random Forest sobre métricas
  01_rgb_grad_cam.ipynb   ZoomNet (CNN sobre RGB) + Grad-CAM
  02_simpleNet.ipynb      SimpleNet (CNN sobre lesión segmentada)
reports/             caso de uso clínico y presentación
docs/img/            figuras usadas en este README
```

---

## 8. Uso

```bash
pip install -r requirements.txt

# 1. Descargar el dataset
#    ejecutar config/01_dowload_data.ipynb

# 2. Generar zoomed/, masks/, lesions/ y los CSV de métricas
python main.py

# 3. Entrenar y evaluar
#    ejecutar los notebooks de notebooks/ en orden
```

---

## 9. Trabajo futuro

Dispositivo portátil basado en Raspberry Pi (cámara y pantalla táctil) que captura
la lesión, estima la probabilidad de melanoma y sincroniza los resultados con una
arquitectura en la nube (almacenamiento, inferencia gestionada, base de datos e
informes) para seguimiento clínico.

![Aplicación futura y arquitectura](docs/img/slide_futuro.jpg)
