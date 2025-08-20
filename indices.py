#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métricas multiclass para segmentación:
Jaccard (macro), Dice (macro), Rand Index (RI), Adjusted Rand Index (ARI),
VOI, GCE, BDE y el Probabilistic Rand Index (PRI) con soporte multi-GT.

Uso:
  python metricas_seg.py GT.png PRED.png
  python metricas_seg.py GT1.png PRED.png --extra_gt GT2.png GT3.png

Requiere: numpy, opencv-python, scikit-learn, scikit-image, scipy
"""

import argparse
import numpy as np
import cv2
from sklearn.metrics import jaccard_score, adjusted_rand_score, rand_score
from skimage.metrics import variation_of_information
from scipy.spatial.distance import directed_hausdorff

# --------------------- Utilidades ---------------------

def load_label_map(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return img.astype(np.int64)

def comb2(n):
    # C(n,2) en float para evitar overflow en divisiones
    return n * (n - 1) / 2.0

def contingency_with_pred(pred_flat, gt_flat, n_pred_labels=None):
    """Devuelve tabla de contingencia (S x L) entre pred y gt,
    junto con los conteos por etiqueta en pred y gt."""
    # id’s contiguos para pred y gt
    pred_vals, pred_inv = np.unique(pred_flat, return_inverse=True)
    if n_pred_labels is None:
        S = pred_vals.size
    else:
        S = n_pred_labels
    gt_vals, gt_inv = np.unique(gt_flat, return_inverse=True)
    L = gt_vals.size
    idx = pred_inv * L + gt_inv
    hist = np.bincount(idx, minlength=S * L).reshape(S, L)
    cnt_pred = np.bincount(pred_inv, minlength=S)
    cnt_gt = np.bincount(gt_inv, minlength=L)
    return hist, cnt_pred, cnt_gt, pred_inv, S, L

# --------------------- Métricas clásicas ---------------------

def jaccard_multiclass(gt, pred):
    labels = np.unique(np.concatenate((gt.flatten(), pred.flatten())))
    return jaccard_score(gt.flatten(), pred.flatten(), labels=labels, average='macro')

def dice_multiclass(gt, pred):
    labels = np.unique(np.concatenate((gt.flatten(), pred.flatten())))
    dice = []
    for lab in labels:
        gt_bin = (gt == lab)
        pr_bin = (pred == lab)
        inter = np.logical_and(gt_bin, pr_bin).sum()
        s = gt_bin.sum() + pr_bin.sum()
        dice.append((2 * inter / s) if s > 0 else 1.0)
    return float(np.mean(dice))

# --------------------- PRI (multi-GT) ---------------------

def probabilistic_rand_index(pred, gts):
    """
    PRI para pred frente a un conjunto de GTs (lista de arrays).
    - Si gts tiene 1 elemento, el resultado coincide con el Rand Index clásico.
    Implementación eficiente basada en combinatorias y tablas de contingencia.
    """
    pred_flat = pred.ravel()
    N = pred_flat.size
    if N < 2:
        return 1.0

    # Ids contiguos para pred (reutilizados en todos los cálculos)
    pred_vals, pred_inv = np.unique(pred_flat, return_inverse=True)
    S = pred_vals.size
    cnt_pred = np.bincount(pred_inv, minlength=S)
    pairs_in_pred = np.sum(cnt_pred * (cnt_pred - 1) / 2.0)

    tot_pairs = comb2(float(N))
    K = len(gts)

    sum_intersections = 0.0     # Σ_k Σ_{s,l} C(|s ∩ l_k|, 2)
    sum_same_gt = 0.0           # Σ_k Σ_l C(n_{k,l}, 2)

    for gt in gts:
        gt_flat = gt.ravel()
        # Histograma 2D pred x gt_k
        gt_vals, gt_inv = np.unique(gt_flat, return_inverse=True)
        L = gt_vals.size
        idx = pred_inv * L + gt_inv
        hist = np.bincount(idx, minlength=S * L).reshape(S, L).astype(np.float64)
        sum_intersections += np.sum(hist * (hist - 1.0) / 2.0)
        cnt_gt = np.bincount(gt_inv, minlength=L).astype(np.float64)
        sum_same_gt += np.sum(cnt_gt * (cnt_gt - 1.0) / 2.0)

    S_same_GT_avg = sum_same_gt / K
    pri = 1.0 - (S_same_GT_avg / tot_pairs) + ((2.0 / K) * sum_intersections - pairs_in_pred) / tot_pairs
    return float(pri)

# --------------------- GCE (implementación fiel) ---------------------

def gce_error(gt, pred):
    """
    Global Consistency Error según LRE:
    GCE(S,S') = (1/N) * min( Σ_i LRE(S,S',i),  Σ_i LRE(S',S,i) )
    con LRE(S,S',i) = (|C_S(i)| - |C_S(i) ∩ C_S'(i)|)/|C_S(i)|
    Implementación vía tabla de contingencia.
    """
    def sum_lre_dir(a, b):
        # filas=a, columnas=b
        a_flat = a.ravel()
        b_flat = b.ravel()
        a_vals, a_inv = np.unique(a_flat, return_inverse=True)
        b_vals, b_inv = np.unique(b_flat, return_inverse=True)
        A = a_vals.size
        B = b_vals.size
        idx = a_inv * B + b_inv
        M = np.bincount(idx, minlength=A * B).reshape(A, B).astype(np.float64)
        size_a = M.sum(axis=1)  # |C_S|
        # Σ_i LRE = N - Σ_s Σ_t M_{s,t}^2 / |C_s|
        term = np.sum((M ** 2).sum(axis=1) / np.where(size_a > 0, size_a, 1.0))
        return a_flat.size - term

    N = gt.size
    dir1 = sum_lre_dir(gt, pred) / N
    dir2 = sum_lre_dir(pred, gt) / N
    return float(min(dir1, dir2))

# --------------------- BDE (aprox. con Hausdorff en bordes) ---------------------

def bde_error(gt, pred):
    edges_gt = cv2.Canny((gt > 0).astype(np.uint8), 100, 200).astype(bool)
    edges_pr = cv2.Canny((pred > 0).astype(np.uint8), 100, 200).astype(bool)
    pts_gt = np.column_stack(np.where(edges_gt))
    pts_pr = np.column_stack(np.where(edges_pr))
    if pts_gt.size == 0 or pts_pr.size == 0:
        return float('nan')
    d1 = directed_hausdorff(pts_gt, pts_pr)[0]
    d2 = directed_hausdorff(pts_pr, pts_gt)[0]
    return float(max(d1, d2))

# --------------------- Main ---------------------

def main(path_gt, path_pred, extra_gts):
    gt = load_label_map(path_gt)
    pred = load_label_map(path_pred)
    if gt.shape != pred.shape:
        raise ValueError("GT y Pred deben tener el mismo tamaño.")

    # Métricas clásicas
    jacc = jaccard_multiclass(gt, pred)
    dice = dice_multiclass(gt, pred)
    ri = rand_score(gt.flatten(), pred.flatten())
    ari = adjusted_rand_score(gt.flatten(), pred.flatten())

    # VOI (puede retornar escalar o tupla según versión)
    voi_val = variation_of_information(gt, pred)
    voi = voi_val[0] if isinstance(voi_val, (tuple, list, np.ndarray)) else voi_val

    # GCE y BDE
    gce = gce_error(gt, pred)
    bde = bde_error(gt, pred)

    # PRI: con una GT equivale al RI; con multi-GT calcula el PRI verdadero
    gts = [gt] + [load_label_map(p) for p in extra_gts]
    pri = probabilistic_rand_index(pred, gts)

    print(f"Jaccard (macro):            {jacc:.6f}")
    print(f"Dice (macro):               {dice:.6f}")
    print(f"Rand Index (RI):            {ri:.6f}")
    print(f"Adjusted Rand Index (ARI):  {ari:.6f}")
    print(f"Probabilistic Rand Index:   {pri:.6f}")
    print(f"Variation of Information:   {float(voi):.6f} (↓ mejor)")
    print(f"Global Consistency Error:   {gce:.6f} (↓ mejor)")
    print(f"Boundary Displacement Err.: {bde:.6f} (↓ mejor)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Métricas multiclass y avanzadas para segmentación (incluye PRI)")
    parser.add_argument("gt", help="Ground truth (etiquetas)")
    parser.add_argument("pred", help="Predicción (etiquetas)")
    parser.add_argument("--extra_gt", nargs="*", default=[], help="GTs adicionales para PRI verdadero (opcional)")
    args = parser.parse_args()
    main(args.gt, args.pred, args.extra_gt)
