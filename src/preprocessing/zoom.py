import cv2

def apply_zoom(image, zoom_factor=0.9):
    """Aplica un zoom centrado a la imagen."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    
    cropped = image[top:top+new_h, left:left+new_w]
    zoomed = cv2.resize(cropped, (w, h))
    return zoomed
