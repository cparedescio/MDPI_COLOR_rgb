#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para calcular el top-hat (white tophat = imagen - apertura) en imágenes de color
con cuatro variantes de ordenamiento en RGB:
  1) "rgb_norm": vectorial en RGB usando un orden reducido por norma (||[R,G,B]||^2) con desempate lexicográfico
  2) "lex": vectorial con orden lexicográfico puro (R, luego G, luego B)
  3) "marginal": por canal (aplica morfología a cada canal y recompone)
  4) "bitmix24": orden total por mezcla de bits 0xRRGGBB (bitmixing de 24 bits)

Guarda las cuatro imágenes resultantes.

Notas:
- El top-hat blanco se define como: TH = I - (I ∘ B), donde ∘ es la apertura (erosión seguida de dilatación).
- Para los casos vectoriales (rgb_norm, lex y bitmix24) la erosión/dilatación eligen el píxel vectorial mínimo/máximo
  bajo el orden correspondiente.
- Implementación pensada para kernels pequeños-medianos (p.ej., 3–11). Para imágenes muy grandes/kernels grandes
  puede tardar; en ese caso considere optimizar con librerías especializadas/Cython.
"""

import argparse
import os
from typing import Tuple
import numpy as np
import cv2

# -------------------- Utilidades --------------------

def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # OpenCV carga en BGR; convertimos a RGB para trabajar consistentemente
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def pad_reflect(color_img: np.ndarray, r: int) -> np.ndarray:
    return np.pad(color_img, ((r, r), (r, r), (0, 0)), mode='reflect')


def sliding_opening_vector(
    img_rgb: np.ndarray,
    ksize: int,
    mode: str = "lex"
) -> np.ndarray:
    """Apertura morfológica (erosión seguida de dilatación) para imagen color
    usando un orden vectorial. Retorna la imagen abierta bajo dicho orden.

    mode:
      - 'lex': orden lexicográfico (R,G,B)
      - 'rgb_norm': orden reducido por norma (R^2+G^2+B^2) con desempate lexicográfico
      - 'bitmix24': orden por entero 24 bits 0xRRGGBB
    """
    assert mode in {"lex", "rgb_norm", "bitmix24"}
    H, W, _ = img_rgb.shape
    r = ksize // 2

    def argmin_pixel(pix: np.ndarray) -> int:
        # pix: (N, 3) uint8
        if mode == 'lex':
            # np.lexsort ordena por la última clave más rápida; para lex (R,G,B), pasamos (B,G,R)
            idx_sorted = np.lexsort((pix[:, 2], pix[:, 1], pix[:, 0]))
            return int(idx_sorted[0])
        elif mode == 'rgb_norm':
            keys = (pix[:, 0].astype(np.int32) ** 2 +
                    pix[:, 1].astype(np.int32) ** 2 +
                    pix[:, 2].astype(np.int32) ** 2)
            # desempate lexicográfico estable
            # obtenemos los mínimos y en empate elegimos por (R,G,B)
            kmin = keys.min()
            mask = (keys == kmin)
            cand = pix[mask]
            if cand.shape[0] == 1:
                # devolver índice global
                return int(np.where(mask)[0][0])
            # ordenar candidatos por lex y tomar el primero
            sub_sorted = np.lexsort((cand[:, 2], cand[:, 1], cand[:, 0]))
            # mapear al índice global original
            return int(np.where(mask)[0][sub_sorted[0]])
        else:  # bitmix24
            keys = (pix[:, 0].astype(np.uint32) << 16) | (pix[:, 1].astype(np.uint32) << 8) | pix[:, 2].astype(np.uint32)
            return int(np.argmin(keys))

    def argmax_pixel(pix: np.ndarray) -> int:
        if mode == 'lex':
            idx_sorted = np.lexsort((pix[:, 2], pix[:, 1], pix[:, 0]))
            return int(idx_sorted[-1])
        elif mode == 'rgb_norm':
            keys = (pix[:, 0].astype(np.int32) ** 2 +
                    pix[:, 1].astype(np.int32) ** 2 +
                    pix[:, 2].astype(np.int32) ** 2)
            kmax = keys.max()
            mask = (keys == kmax)
            cand = pix[mask]
            if cand.shape[0] == 1:
                return int(np.where(mask)[0][0])
            sub_sorted = np.lexsort((cand[:, 2], cand[:, 1], cand[:, 0]))
            return int(np.where(mask)[0][sub_sorted[-1]])
        else:  # bitmix24
            keys = (pix[:, 0].astype(np.uint32) << 16) | (pix[:, 1].astype(np.uint32) << 8) | pix[:, 2].astype(np.uint32)
            return int(np.argmax(keys))

    # Erosión
    padded = pad_reflect(img_rgb, r)
    eroded = np.empty_like(img_rgb)
    for i in range(H):
        for j in range(W):
            block = padded[i:i+ksize, j:j+ksize, :]
            pix = block.reshape(-1, 3)
            eroded[i, j] = pix[argmin_pixel(pix)]

    # Dilatación sobre la erosionada
    padded2 = pad_reflect(eroded, r)
    opened = np.empty_like(img_rgb)
    for i in range(H):
        for j in range(W):
            block = padded2[i:i+ksize, j:j+ksize, :]
            pix = block.reshape(-1, 3)
            opened[i, j] = pix[argmax_pixel(pix)]

    return opened


def opening_marginal(img_rgb: np.ndarray, ksize: int) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    r, g, b = cv2.split(img_rgb)
    r_o = cv2.morphologyEx(r, cv2.MORPH_OPEN, se)
    g_o = cv2.morphologyEx(g, cv2.MORPH_OPEN, se)
    b_o = cv2.morphologyEx(b, cv2.MORPH_OPEN, se)
    return cv2.merge([r_o, g_o, b_o])


def white_tophat(img_rgb: np.ndarray, opened_rgb: np.ndarray) -> np.ndarray:
    # Top-hat blanco: I - apertura (saturado a [0,255])
    th = cv2.subtract(img_rgb, opened_rgb)
    return th


# -------------------- Flujo principal --------------------

def process(input_path: str, ksize: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img = ensure_rgb_uint8(bgr)

    # 1) rgb_norm (orden reducido por norma)
    opened_norm = sliding_opening_vector(img, ksize, mode='rgb_norm')
    th_norm = white_tophat(img, opened_norm)
    cv2.imwrite(os.path.join(outdir, 'tophat_rgb_norm.png'), cv2.cvtColor(th_norm, cv2.COLOR_RGB2BGR))

    # 2) lexicográfico
    opened_lex = sliding_opening_vector(img, ksize, mode='lex')
    th_lex = white_tophat(img, opened_lex)
    cv2.imwrite(os.path.join(outdir, 'tophat_lexicografico.png'), cv2.cvtColor(th_lex, cv2.COLOR_RGB2BGR))

    # 3) marginal (por canal)
    opened_marg = opening_marginal(img, ksize)
    th_marg = white_tophat(img, opened_marg)
    cv2.imwrite(os.path.join(outdir, 'tophat_marginal.png'), cv2.cvtColor(th_marg, cv2.COLOR_RGB2BGR))

    # 4) bitmix 24 bits 0xRRGGBB
    opened_bm = sliding_opening_vector(img, ksize, mode='bitmix24')
    th_bm = white_tophat(img, opened_bm)
    cv2.imwrite(os.path.join(outdir, 'tophat_bitmix24.png'), cv2.cvtColor(th_bm, cv2.COLOR_RGB2BGR))

    # Guardar también la apertura para referencia
    cv2.imwrite(os.path.join(outdir, 'opened_rgb_norm.png'), cv2.cvtColor(opened_norm, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, 'opened_lexicografico.png'), cv2.cvtColor(opened_lex, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, 'opened_marginal.png'), cv2.cvtColor(opened_marg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, 'opened_bitmix24.png'), cv2.cvtColor(opened_bm, cv2.COLOR_RGB2BGR))

    print("Listo. Imágenes guardadas en:", os.path.abspath(outdir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-hat en color (RGB, lexicográfico, marginal, bitmix 24b)")
    parser.add_argument("input", help="Ruta de la imagen de entrada")
    parser.add_argument("--ksize", type=int, default=5, help="Tamaño del elemento estructurante (cuadrado impar, p.ej., 3,5,7,9,...) ")
    parser.add_argument("--outdir", type=str, default="salidas_tophat", help="Carpeta de salida")
    args = parser.parse_args()

    if args.ksize < 1 or args.ksize % 2 == 0:
        raise SystemExit("--ksize debe ser un entero impar >= 1")

    process(args.input, args.ksize, args.outdir)
