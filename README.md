# Detección de Melanomas

El melanoma es el cáncer de piel más grave y difícil de ver a simple vista. Este
proyecto clasifica lunares en **benignos** o **melanoma** a partir de fotos
dermatoscópicas.

## Prueba la aplicación

Está en Hugging Face: **https://huggingface.co/spaces/Martinagg/DermaScan**

Subes una foto del lunar y te dice la probabilidad de melanoma.

---

## Datos

Fotos de lunares etiquetadas como benignas o melanoma.

- **Entrenamiento y test:** ~11.900 y 2.000 imágenes (se descargan con `config/01_dowload_data.ipynb`).
- **Prueba externa:** 45 imágenes del archivo público ISIC, para ver si funciona con fotos de otro origen.

---

## Preprocesado

Cada foto se limpia y se recorta el lunar antes de entrenar
(`src/preprocessing/`, se lanza con `python main.py`):

![Pipeline](docs/img/slide_pipeline.jpg)

1. Zoom al centro y quitar los pelos.
2. Resaltar la zona con pigmento.
3. Separar el lunar del fondo.

De la silueta del lunar se sacan 5 medidas de forma (área, perímetro, lo redondo
que es y dos de simetría), inspiradas en la regla ABCD.

---

## Modelos

| Modelo | Qué mira | Aciertos (test) |
|---|---|---|
| Random Forest | solo las 5 medidas de forma | 72 % |
| ZoomNet | la foto entera del lunar | 89 % |
| SimpleNet | solo el lunar recortado | 81 % |

![Arquitectura de la red](docs/img/arquitectura_cnn.jpg)

**Random Forest** (`notebooks/00_rf_metrics.ipynb`): modelo sencillo que solo usa
los números de forma, sin mirar la imagen. Sirve de referencia. Lo que más pesa
es el perímetro, la circularidad y el área.

<img src="docs/img/rf_feature_importance.png" width="420">

**ZoomNet** (`notebooks/01_rgb_grad_cam.ipynb`): red neuronal grande que mira la
foto completa, con la piel de alrededor. Es la que mejor acierta en el test.

![Curvas ZoomNet](docs/img/zoomnet_curvas.png)

Los mapas de calor (Grad-CAM) confirman que se fija en el lunar, no en el fondo:

![Grad-CAM](docs/img/zoomnet_gradcam.jpg)

**SimpleNet** (`notebooks/02_simpleNet.ipynb`): red más pequeña que solo ve el
lunar recortado. Acierta un poco menos, pero de forma más estable.

![Curvas SimpleNet](docs/img/simplenet_curvas.png)

---

## Resultado con imágenes nuevas (ISIC)

| Modelo | Aciertos | Melanomas detectados | Benignos bien clasificados |
|---|---|---|---|
| ZoomNet | 58 % | 91 % | 23 % |
| SimpleNet | 71 % | 83 % | 59 % |

| ZoomNet | SimpleNet |
|---|---|
| <img src="docs/img/zoomnet_confusion_externo.png" width="360"> | <img src="docs/img/simplenet_confusion_externo.png" width="330"> |

ZoomNet acierta mucho en el test pero falla con imágenes de otro origen: casi
siempre dice "melanoma". SimpleNet, al ver solo el lunar, aguanta mejor.

**Modelo elegido: SimpleNet** (el que está en la demo). Funciona mejor con
imágenes nuevas, entrena estable y es ligero.

---

## Estructura

```
config/              descarga de los datos
src/preprocessing/   limpieza y recorte del lunar
main.py              lanza el preprocesado
exploration/         comparación de las medidas entre benignas y malignas
notebooks/           los tres modelos
reports/             presentación y caso de uso
```

---

## Uso

```bash
pip install -r requirements.txt
# 1. descargar los datos: ejecutar config/01_dowload_data.ipynb
python main.py                 # 2. preprocesar las imágenes
# 3. ejecutar los notebooks de notebooks/
```

---

## Idea a futuro

Un aparato portátil (Raspberry Pi con cámara) que hace la foto, calcula la
probabilidad de melanoma y la envía al médico.

![Aplicación futura](docs/img/slide_futuro.jpg)
