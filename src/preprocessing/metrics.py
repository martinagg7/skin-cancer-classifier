import cv2
import numpy as np
import math

def _leer_mask(mask):
    """Lee una máscara que puede venir como ruta o como array."""
    if isinstance(mask, str):  # si es ruta
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    elif mask.ndim == 3:  # si es RGB/BGR
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask is None:
        raise ValueError("No se pudo leer la máscara.")
    return mask


def calcular_area(mask):
    mask = _leer_mask(mask)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if not np.isfinite(area):
        area = 0.0
    return round(float(area), 2)


def calcular_perimetro(mask):
    mask = _leer_mask(mask)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    perimetro = cv2.arcLength(cnt, True)
    if not np.isfinite(perimetro):
        perimetro = 0.0
    return round(float(perimetro), 2)


def calcular_circularidad(mask):
    mask = _leer_mask(mask)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)

    if perimetro == 0 or area == 0:
        return 0.0

    circ = (4 * math.pi * area) / (perimetro ** 2)

    if not np.isfinite(circ):
        circ = 0.0

    circ = np.clip(circ, 0, 1)
    return round(float(circ), 4)


def calcular_simetria(mask):
    mask = _leer_mask(mask)
    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    if np.sum(mask_bin) == 0:
        return 0.0, 0.0

    y, x = np.where(mask_bin > 0)
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    roi = mask_bin[y_min:y_max+1, x_min:x_max+1]

    h, w = roi.shape
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off, x_off = (size - h)//2, (size - w)//2
    canvas[y_off:y_off+h, x_off:x_off+w] = roi
    mask_centered = canvas

    cy, cx = np.mean(np.column_stack(np.where(mask_centered > 0)), axis=0).astype(int)
    area_total = np.sum(mask_centered)

    if area_total == 0:
        return 0.0, 0.0

    # --- simetría vertical ---
    left = mask_centered[:, :cx]
    right = mask_centered[:, cx:]
    right_flipped = np.fliplr(right)
    min_width = min(left.shape[1], right_flipped.shape[1])
    xor_v = np.logical_xor(left[:, :min_width], right_flipped[:, :min_width])
    sim_v = 1 - (np.sum(xor_v) / area_total)

    # --- simetría horizontal ---
    top = mask_centered[:cy, :]
    bottom = mask_centered[cy:, :]
    bottom_flipped = np.flipud(bottom)
    min_height = min(top.shape[0], bottom_flipped.shape[0])
    xor_h = np.logical_xor(top[:min_height, :], bottom_flipped[:min_height, :])
    sim_h = 1 - (np.sum(xor_h) / area_total)

    sim_v = max(0.0, min(1.0, sim_v))
    sim_h = max(0.0, min(1.0, sim_h))

    return round(sim_v, 3), round(sim_h, 3)