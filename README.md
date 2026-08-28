# Melanoma Detection

Classification of dermoscopic skin lesions as **benign** or **melanoma**. Three
approaches are built and compared: a classic model that works with shape
descriptors of the lesion, and two convolutional networks that receive the image
in different ways.

## Try the app

Deployed on Hugging Face: **https://huggingface.co/spaces/Martinagg/DermaScan**

You upload a photo of the lesion and it returns the probability of melanoma.

---

## 1. Data

Dermoscopic images labeled as benign (`Benign`) or melanoma (`Malignant`).

| Set | Images | Source | Use |
|---|---|---|---|
| Training | ~11,900 | downloaded by `config/01_dowload_data.ipynb` | fitting the models |
| Test | 2,000 (1,000 per class) | same source | evaluation; also used for validation during training |
| External | 45 (22 benign / 23 malignant) | public **ISIC** archive | check generalization to another source |

The `data/` and `models/` folders are not committed.

---

## 2. Preprocessing

Before training, every image goes through a cleanup pipeline
(`src/preprocessing/`, run with `python main.py`) whose goal is to keep only the
lesion.

![Analysis pipeline](docs/img/slide_pipeline.jpg)

| Step | What it does | Why |
|---|---|---|
| Zoom and hair removal | center crop and removal of hairs (black-hat + `inpaint`) | brings the lesion closer and removes hair and the dermatoscope border |
| Saturation channel | work on the S channel of the HSV space | the pigmented area stands out there even with shadows or glare |
| Segmentation | Otsu threshold to separate lesion from background | a clean mask of the lesion |

The result is three versions of each image, one per model:

| Folder | Content | Used by |
|---|---|---|
| `zoomed/` | cropped RGB image | ZoomNet |
| `masks/` | binary mask of the lesion | Random Forest |
| `lesions/` | segmented lesion on black background | SimpleNet |

From the mask, five shape descriptors are also computed, based on the ABCD rule.
They are the input to the Random Forest:

| Descriptor | Definition |
|---|---|
| Area | pixels in the lesion |
| Perimeter | length of the contour |
| Circularity | `4·π·Area / Perimeter²` (1 = circle) |
| Vertical / horizontal symmetry | overlap of the mask with its reflection about each axis |

---

## 3. Models

| Model | Input | Test accuracy | Melanoma recall |
|---|---|---|---|
| Random Forest | 5 shape descriptors | 0.72 | 0.65 |
| **ZoomNet** | zoomed RGB image | **0.89** | 0.86 |
| **SimpleNet** | segmented lesion | 0.81 | 0.70 |

![CNN architecture overview](docs/img/arquitectura_cnn.jpg)

### 3.1. Random Forest (`notebooks/00_rf_metrics.ipynb`)

A classic model that uses only the five shape descriptors, without seeing the
image. It serves as a baseline and is easy to interpret. In
`exploration/00_features.ipynb` you can see that malignant lesions tend to be
less round and less symmetric, though with considerable overlap.

| Feature importance | Confusion matrix (test) |
|---|---|
| <img src="docs/img/rf_feature_importance.png" width="410"> | <img src="docs/img/rf_confusion.png" width="360"> |

It reaches 0.72. Perimeter, circularity and area weigh the most. Shape alone is
not enough: it misses one in three melanomas.

### 3.2. ZoomNet (`notebooks/01_rgb_grad_cam.ipynb`)

A convolutional network that receives the full image of the lesion, with the
surrounding skin.

![ZoomNet training curves](docs/img/zoomnet_curvas.png)

Training and validation move together, no overfitting; AUC 0.958.

<img src="docs/img/zoomnet_roc.png" width="360">

It is the best on the internal test (0.89), but that advantage is lost on images
from another source (section 4). The Grad-CAM maps show that it focuses on the
edge and interior of the lesion, not on the background:

<img src="docs/img/zoomnet_gradcam.jpg" width="760">

### 3.3. SimpleNet (`notebooks/02_simpleNet.ipynb`)

A lighter network that receives only the already cropped lesion, without the
surrounding skin.

![SimpleNet training curves](docs/img/simplenet_curvas.png)

On the internal test it reaches 0.81, slightly below ZoomNet, but with very
stable curves. The notebook also computes its ROC curve.

---

## 4. External evaluation (ISIC)

To check whether the networks learned something useful beyond the training set,
they are tested on 45 images from the ISIC archive, taken with different devices
and under different conditions.

| Model | Accuracy | Melanoma recall | Benign recall |
|---|---|---|---|
| ZoomNet | 0.58 | 0.91 (21/23) | **0.23 (5/22)** |
| SimpleNet | **0.71** | 0.83 (19/23) | 0.59 (13/22) |

| ZoomNet (external) | SimpleNet (external) |
|---|---|
| <img src="docs/img/zoomnet_confusion_externo.png" width="360"> | <img src="docs/img/simplenet_confusion_externo.png" width="330"> |

Here ZoomNet drops to 0.58 and becomes heavily biased toward "malignant": it only
gets 5 of the 22 benign lesions right. Having trained on the full image, it ended
up picking up traits specific to the original source. SimpleNet, which only sees
the cropped lesion, behaves in a much more balanced way and stays at 0.71.

It is only 45 images, so the exact percentages should be taken with caution, but
ZoomNet's drop and its bias toward one class are clear.

---

## 5. Chosen model: SimpleNet

SimpleNet is the model taken to the demo, for three reasons:

- It generalizes better to the external source (0.71 vs 0.58) without always
  leaning toward the same class.
- It trains in a stable way, without overfitting.
- It is lightweight, which makes it easy to use in the demo and in the future
  device.

Random Forest and ZoomNet are kept in the repository because the comparison is
part of the result: they make it clear that lesion shape alone is not enough, and
that a good number on the internal test does not guarantee that the model works
on new data.

---

## 6. Repository structure

```
config/              environment check and data download
src/preprocessing/   zoom, hair removal, segmentation and descriptor computation
main.py              runs the preprocessing on a folder of images
exploration/         analysis of the shape descriptors by class
notebooks/
  00_rf_metrics.ipynb     Random Forest on the descriptors
  01_rgb_grad_cam.ipynb   ZoomNet + Grad-CAM
  02_simpleNet.ipynb      SimpleNet
reports/             clinical use case and presentation
docs/img/            figures used in this README
```

---

## 7. Usage

```bash
pip install -r requirements.txt

# 1. download the data: run config/01_dowload_data.ipynb

# 2. generate zoomed/, masks/, lesions/ and the descriptor CSV
python main.py

# 3. train and evaluate: run the notebooks in order
```

---

## 8. Future work

A portable Raspberry Pi device (camera and screen) that captures the lesion,
estimates the probability of melanoma, and sends the result to a cloud platform
for clinical follow-up.

![Future application and architecture](docs/img/slide_futuro.jpg)
