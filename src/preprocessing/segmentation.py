import cv2
import numpy as np

def segmentar_lesion(rgb, size=(224, 224)):

    # Redimensionar y convertir a HSV
    rgb = cv2.resize(rgb, size)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    S_blur = cv2.GaussianBlur(S, (5, 5), 0)

    # Umbralización con Otsu
    _, mask = cv2.threshold(S_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Selección de la región más grande y centrada
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    center = np.array([w / 2, h / 2])

    valid_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cX, cY = centroids[i]
        dist = np.linalg.norm(np.array([cX, cY]) - center)
        if dist < min(w, h) / 3:
            valid_regions.append((i, area, dist))

    if len(valid_regions) == 0:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        largest_label = max(valid_regions, key=lambda x: x[1])[0]

    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    # Rellenar huecos internos
    inv = cv2.bitwise_not(largest_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv)
    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        if x > 0 and y > 0 and (x + w_box) < w and (y + h_box) < h:
            inv[labels == i] = 0
    filled_mask = cv2.bitwise_not(inv)

    #  Aplicar la máscara al RGB limpio
    lesion_rgb = cv2.bitwise_and(rgb, rgb, mask=filled_mask)

    return filled_mask, lesion_rgb
