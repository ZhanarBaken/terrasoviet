"""
Сегментация карты через Watershed.

Алгоритм:
  1. Тёмные непрерывные линии ЛЮБОГО цвета → границы регионов
  2. Цвет внутри региона → seed для Watershed (помощник на цветных картах)
  3. Watershed → точные полигоны без дробления
  4. OCR кода внутри → поиск в legend_codes.json → название формации
"""

import json
import logging
import os
import re

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MIN_AREA_PX = 1000
_BOUNDARY_DARK = 80        # пиксели темнее этого = граница (любого цвета)
_LEGEND_CODES_PATH = "data/legend_codes.json"

_easyocr_reader = None

def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def segment_map(image: np.ndarray, legend: list[dict]) -> list[dict]:
    """
    Вход:
        image  — BGR карта после crop_border
        legend — из legend.extract_legend() (цветовой помощник)
    Выход:
        [{"name": str, "code": str, "color_hex": str, "contours": [np.ndarray]}]
    """
    legend_codes = _load_legend_codes()

    markers = _create_markers(image, legend)
    cv2.watershed(image.copy(), markers)

    results = _extract_regions(image, markers, legend_codes)
    log.info(f"Watershed: {len(results)} регионов")
    return results


# ── Маркеры ────────────────────────────────────────────────────────────────

def _create_markers(image: np.ndarray, legend: list[dict]) -> np.ndarray:
    """
    Маркеры для watershed:
      0  = неизвестно (зона вдоль границы)
      1  = фон
      2+ = отдельные регионы
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Граница = тёмные пиксели любого цвета
    _, boundary = cv2.threshold(gray, _BOUNDARY_DARK, 255, cv2.THRESH_BINARY_INV)

    # Закрываем мелкие разрывы (дефекты сканирования)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, k, iterations=2)

    region = cv2.bitwise_not(boundary)

    # Цветовой помощник → seeds
    sure_fg = _color_seeds(image, region, legend)

    # Если цвета мало (ч/б карта) → distance transform
    if cv2.countNonZero(sure_fg) < 500:
        dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

    # Неизвестная зона (вдоль границы)
    sure_bg = cv2.dilate(boundary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    return markers


def _color_seeds(image: np.ndarray, region: np.ndarray, legend: list[dict]) -> np.ndarray:
    """
    Цветные карты: seeds = эродированные центры каждого цветового кластера.
    Ч/б карты: пустая маска → fallback на distance transform.
    """
    sure_fg = np.zeros(image.shape[:2], dtype=np.uint8)
    if not legend:
        return sure_fg

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for entry in legend:
        lower = np.array(entry["hsv_lower"], dtype=np.uint8)
        upper = np.array(entry["hsv_upper"], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_and(mask, region)   # только внутри регионов

        # Берём только центр каждого пятна как seed
        eroded = cv2.erode(mask, k, iterations=2)
        sure_fg = cv2.bitwise_or(sure_fg, eroded)

    return sure_fg


# ── Извлечение регионов ────────────────────────────────────────────────────

def _extract_regions(image: np.ndarray, markers: np.ndarray, legend_codes: dict) -> list[dict]:
    results = []

    for label in np.unique(markers):
        if label <= 1:   # 0 = watershed граница, 1 = фон
            continue

        mask = np.uint8(markers == label) * 255

        if cv2.countNonZero(mask) < _MIN_AREA_PX:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        code = _ocr_region(image, mask)
        name = _lookup_code(code, legend_codes) if code else ""
        color_hex = _dominant_color(image, mask)

        if not name:
            name = code if code else color_hex

        results.append({
            "name": name,
            "code": code,
            "color_hex": color_hex,
            "contours": [cv2.approxPolyDP(c, 3, True) for c in contours],
        })

    return results


# ── OCR ───────────────────────────────────────────────────────────────────

def _ocr_region(image: np.ndarray, mask: np.ndarray) -> str:
    """OCR геологического кода внутри региона (EasyOCR)."""
    try:
        reader = _get_reader()
    except Exception:
        return ""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""

    x, y, w, h = cv2.boundingRect(contours[0])
    if w < 30 or h < 20:
        return ""

    roi = image[y:y + h, x:x + w].copy()
    roi[mask[y:y + h, x:x + w] == 0] = 255

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Масштабируем мелкие надписи
    scale = max(1, 60 // max(h, 1) + 1)
    if scale > 1:
        binary = cv2.resize(binary, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)

    try:
        results = reader.readtext(binary, detail=0, paragraph=False)
        text = " ".join(results).strip()
    except Exception:
        return ""

    match = re.search(r'[A-Za-zА-Яа-яοεδγβα][A-Za-z0-9]*', text)
    return match.group(0) if match else ""


# ── Поиск в легенде ───────────────────────────────────────────────────────

def _lookup_code(code: str, legend_codes: dict) -> str:
    """Точный поиск + fuzzy matching (edit distance ≤ 2)."""
    if not code or not legend_codes:
        return ""

    norm = _normalize(code)

    for section in legend_codes.values():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key.startswith("_"):
                continue
            if _normalize(key) == norm:
                return value

    best, best_d = "", float("inf")
    for section in legend_codes.values():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key.startswith("_"):
                continue
            d = _edit_distance(norm, _normalize(key))
            if d < best_d and d <= 2:
                best_d, best = d, value

    return best


def _normalize(code: str) -> str:
    """H²ₘ → hm2, C₁ → c1"""
    t = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
    return code.translate(t).replace(" ", "").lower()


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(dp[j], dp[j-1], prev)
    return dp[n]


# ── Вспомогательные ───────────────────────────────────────────────────────

def _dominant_color(image: np.ndarray, mask: np.ndarray) -> str:
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return "#000000"
    m = np.median(pixels, axis=0).astype(int)
    return "#{:02x}{:02x}{:02x}".format(m[2], m[1], m[0])


def _load_legend_codes() -> dict:
    if not os.path.exists(_LEGEND_CODES_PATH):
        log.warning(f"legend_codes.json не найден: {_LEGEND_CODES_PATH}")
        return {}
    with open(_LEGEND_CODES_PATH, encoding="utf-8") as f:
        return json.load(f)
