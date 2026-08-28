# Detección de Melanomas

El melanoma es el cáncer de piel más grave y difícil de detectar a simple vista.
Los dermatólogos se guían por la regla **ABCD** (Asimetría, Bordes, Color,
Diámetro), pero aun así hay melanomas que pasan desapercibidos.

Este proyecto clasifica lunares en **benignos** o **melanoma** a partir de fotos
dermatoscópicas y compara tres formas de hacerlo.

## Prueba la aplicación

Está desplegada en Hugging Face: **https://huggingface.co/spaces/Martinagg/DermaScan**

Subes una foto del lunar y devuelve la probabilidad de melanoma.

---

## 1. Datos

Fotos dermatoscópicas etiquetadas como benignas (`Benign`) o melanoma
(`Malignant`).

| Conjunto | Imágenes | De dónde salen | Para qué |
|---|---|---|---|
| Entrenamiento | ~11.900 | se descargan con `config/01_dowload_data.ipynb` | ajustar los modelos |
| Test | 2.000 (1.000 por clase) | misma fuente | medir cómo de bien clasifican |
| Externo | 45 (22 benignas / 23 malignas) | archivo público **ISIC** | ver si funcionan con fotos de otro origen |

Las carpetas `data/` y `models/` no se suben al repositorio.

---

## 2. Preprocesado

Antes de entrenar, cada foto pasa por una limpieza para quedarnos solo con el
lunar. Está en `src/preprocessing/` y se lanza con `python main.py`.

![Pipeline de análisis](docs/img/slide_pipeline.jpg)

| Paso | Qué se hace | Por qué |
|---|---|---|
| **Zoom y quitar pelos** | se recorta el 90 % central de la imagen y se borran los pelos con un filtro (*black-hat* + *inpaint*, que detecta líneas finas oscuras y las rellena con el color de alrededor) | acerca el lunar y elimina el vello y el borde negro del dermatoscopio, que estorban |
| **Canal de saturación** | se pasa la imagen a color HSV y se usa solo el canal *S* (cuánto color tiene cada píxel) | la zona con pigmento resalta mucho en ese canal, aunque la foto tenga sombras o brillos |
| **Umbral de Otsu** | Otsu busca automáticamente el valor de gris que mejor separa "lunar" de "piel" y crea una máscara en blanco y negro; luego se coge la mancha central y se rellenan los huecos | deja una silueta limpia del lunar |

De ahí salen tres versiones de cada imagen, una para cada modelo:

| Carpeta | Qué contiene | La usa |
|---|---|---|
| `zoomed/` | la foto a color recortada | ZoomNet |
| `masks/` | la silueta del lunar en blanco y negro | Random Forest |
| `lesions/` | el lunar recortado sobre fondo negro | SimpleNet |

### Medidas de forma

De la silueta se calculan cinco números, inspirados en la regla ABCD. Son la
entrada del Random Forest.

| Medida | Qué mide |
|---|---|
| Área | tamaño del lunar en píxeles |
| Perímetro | longitud del borde; crece cuando el borde es irregular |
| Circularidad (`4·π·Área / Perímetro²`) | cómo de redondo es (1 = círculo perfecto) |
| Simetría vertical | parecido entre la mitad izquierda y la derecha |
| Simetría horizontal | parecido entre la mitad de arriba y la de abajo |

---

## 3. Modelos

| Modelo | Qué mira | Técnica | Aciertos (test) | Melanomas detectados |
|---|---|---|---|---|
| Random Forest | las 5 medidas de forma | árboles de decisión | 72 % | 65 % |
| **ZoomNet** | la foto entera del lunar | red neuronal convolucional (grande) | **89 %** | 86 % |
| **SimpleNet** | solo el lunar recortado | red neuronal convolucional (pequeña) | 81 % | 70 % |

Las dos redes son **convolucionales (CNN)**: aplican filtros que recorren la
imagen y detectan primero bordes y colores, luego formas, y al final deciden
entre benigno y melanoma. Se diferencian en el tamaño.

![Esquema de la red](docs/img/arquitectura_cnn.jpg)

### 3.1. Random Forest (`notebooks/00_rf_metrics.ipynb`)

Un **Random Forest** entrena muchos árboles de decisión (aquí 400), cada uno con
una parte de los datos, y clasifica por votación de todos. Solo usa los cinco
números de forma, no la imagen. Sirve de referencia sencilla y fácil de explicar.

Antes se comparan las medidas entre lesiones benignas y malignas
(`exploration/00_features.ipynb`): las malignas suelen ser menos redondas y menos
simétricas.

| Peso de cada medida | Aciertos y fallos (test) |
|---|---|
| <img src="docs/img/rf_feature_importance.png" width="410"> | <img src="docs/img/rf_confusion.png" width="360"> |

Acierta el **72 %**. Lo que más pesa es el perímetro, la circularidad y el área.
Se queda corto: no detecta 35 de cada 100 melanomas.

### 3.2. ZoomNet: red sobre la foto completa (`notebooks/01_rgb_grad_cam.ipynb`)

Red convolucional **grande** (cuatro bloques de filtros) que mira la foto entera
del lunar con la piel de alrededor. Se entrena con el algoritmo *Adam* y se para
sola cuando deja de mejorar (época 23).

En las curvas, entrenamiento y validación van juntas: la red aprende sin
**memorizar** (sin sobreajuste).

![Curvas de entrenamiento de ZoomNet](docs/img/zoomnet_curvas.png)

La **curva ROC** resume el equilibrio entre aciertos y falsas alarmas para todos
los umbrales posibles; el área bajo la curva (AUC) es 0.96 (1 sería perfecto).

<img src="docs/img/zoomnet_roc.png" width="360">

Es la que **mejor acierta en el test interno (89 %)**, pero esa ventaja no se
mantiene con imágenes de otro origen (sección 4).

**Grad-CAM** es una técnica que pinta sobre la imagen las zonas que la red ha
usado para decidir. Aquí confirma que mira el borde y el interior del lunar, no
el fondo:

<img src="docs/img/zoomnet_gradcam.jpg" width="760">

### 3.3. SimpleNet: red sobre el lunar recortado (`notebooks/02_simpleNet.ipynb`)

Red convolucional **pequeña** (tres bloques) que solo ve el lunar ya recortado,
sin la piel de alrededor, para que no pueda fijarse en cosas que no son la
lesión. Al tener menos parámetros, le cuesta más memorizar.

![Curvas de entrenamiento de SimpleNet](docs/img/simplenet_curvas.png)

Acierta el **81 %** en el test interno, algo menos que ZoomNet, pero con curvas
muy estables. El notebook calcula también su curva ROC.

---

## 4. Resultado con imágenes nuevas (ISIC)

Se prueban las dos redes con 45 fotos de ISIC, un origen distinto al del
entrenamiento. Es la prueba de si han aprendido a reconocer lesiones o solo a
reconocer *ese* conjunto de fotos.

| Modelo | Aciertos | Melanomas detectados | Benignos bien clasificados |
|---|---|---|---|
| ZoomNet | 58 % | 91 % (21/23) | **23 % (5/22)** |
| SimpleNet | **71 %** | 83 % (19/23) | 59 % (13/22) |

| ZoomNet (externo) | SimpleNet (externo) |
|---|---|
| <img src="docs/img/zoomnet_confusion_externo.png" width="360"> | <img src="docs/img/simplenet_confusion_externo.png" width="330"> |

ZoomNet baja al 58 % y casi siempre dice "melanoma" (solo acierta 5 de 22 lunares
benignos): al ver la imagen entera, se ha acostumbrado a detalles propios de las
fotos con las que se entrenó. SimpleNet, que solo ve el lunar recortado, se
mantiene equilibrada (71 %) con imágenes nuevas.

---

## 5. Modelo elegido: SimpleNet

Es el que está en la demo. Razones:

- **Funciona mejor con imágenes nuevas** (71 % frente al 58 % de ZoomNet) y sin
  decantarse siempre por la misma clase.
- **Entrena de forma estable**, sin memorizar.
- **Es ligero**, fácil de usar en la demo y en el dispositivo futuro.

Random Forest y ZoomNet se dejan en el repositorio para la comparación: muestran
que la forma por sí sola no basta y que acertar en el test interno no garantiza
acertar fuera.

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

# 1. descargar los datos: ejecutar config/01_dowload_data.ipynb

# 2. generar las imágenes procesadas y el CSV con las medidas
python main.py

# 3. entrenar y evaluar: ejecutar los notebooks de notebooks/ en orden
```

---

## 8. Idea a futuro

Un aparato portátil basado en Raspberry Pi (con cámara y pantalla) que hace la
foto del lunar, calcula la probabilidad de melanoma y envía el resultado al
médico para el seguimiento.

![Aplicación futura y arquitectura](docs/img/slide_futuro.jpg)
