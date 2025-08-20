import os
import cv2
import numpy as np
from collections import defaultdict

# ---------- Utilidades ----------
def ensure_same_size(a, b):
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    return a, b

def mse_rgb(img1, img2):
    img1, img2 = ensure_same_size(img1, img2)
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse_total = np.mean(diff**2)
    mse_r = np.mean(diff[:,:,2]**2)  # OpenCV es BGR
    mse_g = np.mean(diff[:,:,1]**2)
    mse_b = np.mean(diff[:,:,0]**2)
    return mse_total, (mse_r, mse_g, mse_b)

def dice_binary(a, b):
    # a y b: máscaras booleanas o {0,1}
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    size = a.sum() + b.sum()
    return (2.0 * inter) / size if size > 0 else 1.0  # si ambas vacías: 1.0

def jaccard_binary(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter / union) if union > 0 else 1.0

# ---------- Caso 1: MSE en imágenes RGB ----------
def compare_mse_rgb(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise ValueError("No se pudieron leer una o ambas imágenes.")
    img1, img2 = ensure_same_size(img1, img2)
    mse_total, (mse_r, mse_g, mse_b) = mse_rgb(img1, img2)
    return {
        "MSE_total": float(mse_total),
        "MSE_R": float(mse_r),
        "MSE_G": float(mse_g),
        "MSE_B": float(mse_b)
    }

# ---------- Caso 2: Segmentación binaria ----------
def compare_binary_masks(path_mask1, path_mask2, threshold=127):
    # Lee en escala de grises y umbraliza a {0,1}
    m1 = cv2.imread(path_mask1, cv2.IMREAD_GRAYSCALE)
    m2 = cv2.imread(path_mask2, cv2.IMREAD_GRAYSCALE)
    if m1 is None or m2 is None:
        raise ValueError("No se pudieron leer una o ambas máscaras.")
    m1, m2 = ensure_same_size(m1, m2)
    a = (m1 > threshold).astype(np.uint8)
    b = (m2 > threshold).astype(np.uint8)
    d = dice_binary(a, b)
    j = jaccard_binary(a, b)
    return {"Dice": float(d), "Jaccard": float(j)}

# ---------- Caso 3: Segmentación multiclase codificada por color ----------
def image_colors_to_labels(img):
    """Convierte cada color BGR único en una etiqueta entera."""
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3)
    # unique devuelve orden estable; mapeamos color->id
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    labels = inv.reshape(h, w).astype(np.int32)
    # Diccionario (tuple(B,G,R)) -> label_id
    palette = {tuple(c.tolist()): i for i, c in enumerate(colors)}
    return labels, palette

def per_class_metrics_multiclass(path_labimg1, path_labimg2):
    img1 = cv2.imread(path_labimg1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path_labimg2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise ValueError("No se pudieron leer una o ambas imágenes.")
    img1, img2 = ensure_same_size(img1, img2)

    lab1, pal1 = image_colors_to_labels(img1)
    lab2, pal2 = image_colors_to_labels(img2)

    # Conjunto total de etiquetas presentes en cualquiera
    classes = np.union1d(np.unique(lab1), np.unique(lab2))

    per_class = {}
    tp_sum = fp_sum = fn_sum = 0

    for k in classes:
        a = (lab1 == k)
        b = (lab2 == k)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        sa = a.sum(); sb = b.sum()

        dice = (2*inter)/(sa+sb) if (sa+sb)>0 else 1.0
        jacc = (inter/union) if union>0 else 1.0

        # Para micro-promedio: TP/FP/FN por clase (1-vs-rest)
        tp_sum += inter
        fp_sum += (sb - inter)
        fn_sum += (sa - inter)

        per_class[int(k)] = {"Dice": float(dice), "Jaccard": float(jacc),
                             "Pixels_A": int(sa), "Pixels_B": int(sb)}

    # Macro: promedio simple por clases presentes
    macro_dice = float(np.mean([m["Dice"] for m in per_class.values()])) if per_class else 1.0
    macro_jacc = float(np.mean([m["Jaccard"] for m in per_class.values()])) if per_class else 1.0

    # Micro: con TP, FP, FN agregados
    micro_dice = (2*tp_sum)/(2*tp_sum + fp_sum + fn_sum) if (2*tp_sum + fp_sum + fn_sum)>0 else 1.0
    micro_jacc = (tp_sum)/(tp_sum + fp_sum + fn_sum) if (tp_sum + fp_sum + fn_sum)>0 else 1.0

    return {
        "per_class": per_class,
        "macro": {"Dice": macro_dice, "Jaccard": macro_jacc},
        "micro": {"Dice": float(micro_dice), "Jaccard": float(micro_jacc)}
    }

# ----------------- EJEMPLOS DE USO -----------------
if __name__ == "__main__":
    # Rutas de ejemplo:
    # 1) MSE en RGB:
    
    for i in range(1, 13):
    # Genera el nombre con dos dígitos (01, 02, ..., 10)     
        img1=f"Orig-{i:02d}_MnGt_b-e.bmp"
        img2=f"Orig-{i:02d}gt-e.bmp"
        
        # Verifica que el archivo exista antes de comparar
        if os.path.exists(img1) and os.path.exists(img2):
                print(f"\nComparando {img1} con {img2}")
                # Ejemplo usando MSE:
                print(compare_mse_rgb(img1, img2))
                print(compare_binary_masks(img1, img2, threshold=127))
        else:
                print(f"⚠️ No se encontró {img1} o {img2}")
    pass
