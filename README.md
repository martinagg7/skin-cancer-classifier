# Melanoma Detection

Classification of dermoscopic skin lesions as **benign** or **melanoma**. Three
approaches are built and compared: a classic model that works with shape
descriptors of the lesion, and two convolutional networks that receive the image
in different ways.


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
fed to the Random Forest:

| Descriptor | What it measures |
|---|---|
| Area | pixels in the lesion |
| Perimeter | length of the contour |
| Circularity | `4·π·Area / Perimeter²` (1 = perfect circle) |
| Vertical / horizontal symmetry | how well the mask overlaps its reflection about each axis |

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




