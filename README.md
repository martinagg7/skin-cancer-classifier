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

## 1. Datos

Imágenes dermatoscópicas etiquetadas como benignas (`Benign`) o melanoma
(`Malignant`).

| Conjunto | Imágenes | De dónde salen | Para qué |
|---|---|---|---|
| Entrenamiento | ~11.900 | se descargan con `config/01_dowload_data.ipynb` | ajustar los modelos |
| Test | 2.000 (1.000 por clase) | misma fuente | evaluar y validar durante el entrenamiento |
| Externo | 45 (22 benignas / 23 malignas) | archivo público **ISIC** | comprobar si funcionan con imágenes de otro origen |

Las carpetas `data/` y `models/` no se suben al repositorio.

---

## 2. Preprocesado y características

Todo el preprocesado está en `src/preprocessing/` y se lanza con `python main.py`.

![Pipeline de análisis](docs/img/slide_pipeline.jpg)

| Paso | Qué hace | Por qué |
|---|---|---|
| **Zoom + limpieza** | recorte central al 90 % y eliminación de pelos (black-hat + `inpaint`) | centra la lesión y quita bordes del dermatoscopio y vello que la tapan |
| **Espacio HSV** | se coge el canal de saturación (S) y se suaviza | marca bien la zona con pigmento aunque la imagen tenga ruido |
| **Máscara final** | umbral de Otsu, se coge la mancha central y se rellenan huecos | separa la lesión del fondo |

De aquí salen tres versiones de cada imagen, cada una para un modelo:

| Salida | Qué es | La usa |
|---|---|---|
| `zoomed/` | imagen a color recortada | ZoomNet |
| `masks/` | silueta del lunar en blanco y negro | Random Forest (a través de las medidas) |
| `lesions/` | el lunar recortado sobre fondo negro | SimpleNet |

### Características de forma

De la silueta del lunar se calculan cinco números, inspirados en la regla ABCD:

| Característica | Qué mide |
|---|---|
| Área | tamaño del lunar (en píxeles) |
| Perímetro | longitud del borde; es mayor cuando el borde es irregular |
| Circularidad (`4·π·A / P²`) | cómo de redondo es (1 = círculo perfecto) |
| Simetría vertical | parecido entre la mitad izquierda y la derecha |
| Simetría horizontal | parecido entre la mitad de arriba y la de abajo |

Se guardan en un CSV y son la entrada del Random Forest.

---

## 3. Modelos planteados

| Modelo | Qué mira | Idea | Aciertos (test) | Melanomas detectados |
|---|---|---|---|---|
| Random Forest | las 5 medidas de forma | punto de partida, fácil de interpretar | 72 % | 65 % |
| **ZoomNet** | la foto entera del lunar | red neuronal que ve la lesión y la piel de alrededor | **89 %** | 86 % |
| **SimpleNet** | solo el lunar recortado | red neuronal más pequeña | 81 % | 70 % |

Las dos redes tienen la misma idea: varias capas que van detectando patrones cada
vez más complejos en la imagen y, al final, deciden entre benigno y melanoma.
ZoomNet es más grande; SimpleNet, más pequeña.

![Arquitectura de la red](docs/img/arquitectura_cnn.jpg)

### 3.1. Random Forest (`notebooks/00_rf_metrics.ipynb`)

Solo usa los cinco números de forma del lunar, sin mirar la imagen. Sirve para ver
cuánto se puede acertar únicamente con la forma.

Antes se comparan esas medidas entre lesiones benignas y malignas
(`exploration/00_features.ipynb`): las malignas suelen ser menos redondas y menos
simétricas.

| Peso de cada medida | Aciertos y fallos (test) |
|---|---|
| <img src="docs/img/rf_feature_importance.png" width="420"> | <img src="docs/img/rf_confusion.png" width="360"> |

Acierta el **72 %**. Las medidas que más pesan son el perímetro, la circularidad y
el área. Se queda corto: no detecta 35 de cada 100 melanomas.

### 3.2. ZoomNet: red sobre la foto completa (`notebooks/01_rgb_grad_cam.ipynb`)

Mira la foto entera del lunar (a color), con la piel de alrededor. Es la red más
grande de las dos. Se entrena hasta que deja de mejorar (para sola en la
época 23).

**Curvas de entrenamiento** (salen del notebook):

![Curvas de entrenamiento de ZoomNet](docs/img/zoomnet_curvas.png)

**Curva ROC (test):**

<img src="docs/img/zoomnet_roc.png" width="360">

Es la que **mejor acierta en el test interno**: 89 %, con las curvas de
entrenamiento y validación juntas (no memoriza). Pero esa ventaja no se mantiene
con imágenes de otro origen (sección 4).

**Grad-CAM.** Estos mapas de calor muestran en qué parte de la imagen se fija la
red para decidir. Se centra en el borde y el interior del lunar, no en el fondo:

<img src="docs/img/zoomnet_gradcam.jpg" width="760">

### 3.3. SimpleNet: red sobre el lunar recortado (`notebooks/02_simpleNet.ipynb`)

Solo ve el lunar ya recortado, sin la piel de alrededor, para que no pueda
fijarse en cosas que no son la lesión. Es más pequeña que ZoomNet.

**Curvas de entrenamiento** (salen del notebook):

![Curvas de entrenamiento de SimpleNet](docs/img/simplenet_curvas.png)

Acierta el **81 %** en el test interno, algo menos que ZoomNet, pero sus curvas
son muy estables. El notebook calcula también su curva ROC.

---

## 4. Evaluación en datos externos (ISIC)

Prueba con 45 imágenes de ISIC, un origen distinto al de entrenamiento.

| Modelo | Aciertos | Melanomas detectados | Benignos bien clasificados |
|---|---|---|---|
| ZoomNet | 58 % | 91 % (21/23) | **23 % (5/22)** |
| SimpleNet | **71 %** | 83 % (19/23) | 59 % (13/22) |

| ZoomNet (externo) | SimpleNet (externo) |
|---|---|
| <img src="docs/img/zoomnet_confusion_externo.png" width="360"> | <img src="docs/img/simplenet_confusion_externo.png" width="330"> |

ZoomNet, la mejor en el test interno, baja al 58 % y casi siempre dice "melanoma"
(solo acierta 5 de 22 lunares benignos): al ver la imagen entera, se ha
acostumbrado a detalles propios de las fotos con las que se entrenó. SimpleNet,
que solo ve el lunar recortado, se mantiene equilibrada (71 %) con imágenes
nuevas.

---

## 5. Modelo elegido: SimpleNet

Se despliega **SimpleNet** en la demo [DermaScan](https://huggingface.co/spaces/Martinagg/DermaScan):

- **Funciona mejor con imágenes nuevas.** Pasa de 58 % (ZoomNet) a 71 % de
  aciertos en el conjunto externo, y sin decantarse siempre por la misma clase.
- **Entrenamiento estable**, sin memorizar los datos.
- **Es más ligera**, lo que facilita usarla en la demo y en el dispositivo futuro.

ZoomNet y el Random Forest se dejan en el repositorio para la comparación:
enseñan, cada uno, que acertar en el test interno no garantiza acertar fuera y
que la forma por sí sola no basta.

---

## 6. Estructura del repositorio

```
config/              comprobación del entorno y descarga de los datos
src/preprocessing/   zoom, limpieza, recorte del lunar y cálculo de las medidas
main.py              lanza el preprocesado sobre una carpeta de imágenes
exploration/         comparación de las medidas de forma entre benignas y malignas
notebooks/
  00_rf_metrics.ipynb     Random Forest sobre las medidas de forma
  01_rgb_grad_cam.ipynb   ZoomNet (red sobre la foto completa) + Grad-CAM
  02_simpleNet.ipynb      SimpleNet (red sobre el lunar recortado)
reports/             caso de uso clínico y presentación
docs/img/            figuras usadas en este README
```

---

## 7. Uso

```bash
pip install -r requirements.txt

# 1. Descargar el dataset
#    ejecutar config/01_dowload_data.ipynb

# 2. Generar las imágenes procesadas y el CSV con las medidas
python main.py

# 3. Entrenar y evaluar
#    ejecutar los notebooks de notebooks/ en orden
```

---

## 8. Trabajo futuro

Dispositivo portátil basado en Raspberry Pi (cámara y pantalla táctil) que captura
la lesión, estima la probabilidad de melanoma y sincroniza los resultados con una
arquitectura en la nube (almacenamiento, inferencia gestionada, base de datos e
informes) para seguimiento clínico.

![Aplicación futura y arquitectura](docs/img/slide_futuro.jpg)
