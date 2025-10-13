import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def calcular_area(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.nan

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    return round(area, 2)


def calcular_perimetro(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.nan

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan

    cnt = max(contours, key=cv2.contourArea)
    perimetro = cv2.arcLength(cnt, True)
    return round(perimetro, 2)


def calcular_circularidad(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.nan

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)

    if perimetro == 0:
        return 0.0

    circularidad = (4 * math.pi * area) / (perimetro ** 2)
    return round(circularidad, 4)


def calcular_simetria(mask_path, visualizar=False):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.nan, np.nan

    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    if np.sum(mask_bin) == 0:
        return np.nan, np.nan

    # Centrar la máscara
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

    coords = np.column_stack(np.where(mask_centered > 0))
    cy, cx = np.mean(coords, axis=0).astype(int)
    area_total = np.sum(mask_centered)

    # Vertical
    left = mask_centered[:, :cx]
    right = mask_centered[:, cx:]
    right_flipped = np.fliplr(right)
    min_width = min(left.shape[1], right_flipped.shape[1])
    xor_v = np.logical_xor(left[:, :min_width], right_flipped[:, :min_width])
    sim_v = 1 - (np.sum(xor_v) / area_total)

    # Horizontal
    top = mask_centered[:cy, :]
    bottom = mask_centered[cy:, :]
    bottom_flipped = np.flipud(bottom)
    min_height = min(top.shape[0], bottom_flipped.shape[0])
    xor_h = np.logical_xor(top[:min_height, :], bottom_flipped[:min_height, :])
    sim_h = 1 - (np.sum(xor_h) / area_total)

    if visualizar:
        plt.imshow(mask_centered, cmap='gray')
        plt.axvline(cx, color='r', linestyle='--')
        plt.axhline(cy, color='b', linestyle='--')
        plt.title(f"Simetría V={sim_v:.2f}  H={sim_h:.2f}")
        plt.show()

    return round(sim_v, 4), round(sim_h, 4)