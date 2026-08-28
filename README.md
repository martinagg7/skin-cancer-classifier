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

Training uses about 11,900 images, with a balanced test set of 2,000 (1,000 per
class); both are downloaded by `config/01_dowload_data.ipynb`, and the test set
also acts as validation during training. Generalization is checked on a separate
set of 45 images (22 benign, 23 malignant) from the public **ISIC** archive.

The `data/` and `models/` folders are not committed.

---

## 2. Preprocessing

Before training, every image goes through a cleanup pipeline
(`src/preprocessing/`, run with `python main.py`) that keeps only the lesion:

1. **Zoom and hair removal.** A center crop brings the lesion closer, and the
   hairs and the dark dermatoscope border are removed with a black-hat filter
   plus `inpaint`.
2. **Saturation channel.** The image is converted to HSV and only the S channel
   is used, where the pigmented area stands out even with shadows or glare.
3. **Segmentation.** An Otsu threshold separates lesion from background, giving a
   clean mask.

![Analysis pipeline](docs/img/slide_pipeline.jpg)

This produces three versions of each image: the cropped RGB photo (`zoomed/`,
used by ZoomNet), the binary mask (`masks/`), and the segmented lesion on a black
background (`lesions/`, used by SimpleNet).

From the mask, five shape descriptors are computed (based on the ABCD rule) and
fed to the Random Forest: **area** (lesion pixels), **perimeter** (contour
length), **circularity** (`4·π·Area / Perimeter²`, where 1 is a circle), and
**vertical and horizontal symmetry** (how well the mask overlaps its reflection
about each axis).

---

## 3. Models

Three models, ordered by test accuracy:

- **Random Forest** — input: the 5 shape descriptors — 0.72 accuracy, 0.65 melanoma recall
- **ZoomNet** — input: the zoomed RGB image — 0.89 accuracy, 0.86 melanoma recall
- **SimpleNet** — input: the segmented lesion — 0.81 accuracy, 0.70 melanoma recall

![CNN architecture overview](docs/img/arquitectura_cnn.jpg)

### 3.1. Random Forest (`notebooks/00_rf_metrics.ipynb`)

A classic model that uses only the five shape descriptors, without seeing the
image. It serves as a baseline and is easy to interpret. In
`exploration/00_features.ipynb` you can see that malignant lesions tend to be
less round and less symmetric, though with considerable overlap.

<img src="docs/img/rf_feature_importance.png" width="440">

<img src="docs/img/rf_confusion.png" width="380">

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

- **ZoomNet** — 0.58 accuracy — catches 21 of 23 melanomas but only 5 of 22 benign lesions
- **SimpleNet** — 0.71 accuracy — 19 of 23 melanomas, 13 of 22 benign lesions

<img src="docs/img/zoomnet_confusion_externo.png" width="380">

<img src="docs/img/simplenet_confusion_externo.png" width="360">

ZoomNet collapses to 0.58 and labels almost everything "malignant" (only 5 of 22
benign lesions correct): trained on the full image, it learned traits specific to
the training source. SimpleNet, seeing only the cropped lesion, stays balanced at
0.71. With just 45 images the exact numbers are noisy, but the gap is clear.

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
