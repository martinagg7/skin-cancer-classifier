import cv2
import numpy as np

def quitar_pelos(rgb):
    """Elimina pelos de una imagen RGB usando black-hat + inpaint."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    rgb_clean = cv2.inpaint(rgb, hair_mask, 3, cv2.INPAINT_TELEA)
    
    return rgb_clean
