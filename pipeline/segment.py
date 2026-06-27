"""
Сегментация карты — модуль ML-участника (фиолетовый).

Классификация ведётся по ДВУМ критериям:
  1. Цвет (HSV inRange по палитре легенды)
  2. Текстовые коды внутри региона (OCR: C₁, m₃, ε₁ и т.д.)

Если код OCR совпадает с кодом из легенды → перезаписывает классификацию по цвету.
"""

import logging
import re

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MIN_AREA_PX = 200   # минимальная площадь полигона (пиксели)
_APPROX_EPS = 3      # упрощение контура (пикселей)


def segment_map(image: np.ndarray, legend: list[dict]) -> list[dict]:
    """
    Вход:
        image  — BGR карта после crop_border (из georeference.build_transform)
        legend — список из legend.extract_legend()
    Выход:
        [{
            "name":      str,           # название/код формации
            "color_hex": str,           # hex цвет
            "contours":  [np.ndarray],  # пиксельные контуры
        }]
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    results = []

    for entry in legend:
        lower = np.array(entry["hsv_lower"], dtype=np.uint8)
        upper = np.array(entry["hsv_upper"], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = _clean_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = _filter_and_simplify(contours)

        if not valid:
            continue

        # OCR внутри каждого контура → уточнение классификации
        name = entry["name"]
        for c in valid:
            ocr_code = _ocr_region(image, c)
            if ocr_code and entry.get("code") and _codes_match(ocr_code, entry["code"]):
                name = entry["name"]   # подтверждение совпадения
            elif ocr_code and not entry.get("code"):
                name = ocr_code        # нет кода в легенде → используем OCR

        results.append({
            "name": name,
            "color_hex": entry["color_hex"],
            "contours": valid,
        })

    log.info(f"Сегментация: {len(results)} формаций, "
             f"{sum(len(r['contours']) for r in results)} полигонов")
    return results


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _filter_and_simplify(contours) -> list[np.ndarray]:
    result = []
    for c in contours:
        if cv2.contourArea(c) < _MIN_AREA_PX:
            continue
        simplified = cv2.approxPolyDP(c, _APPROX_EPS, closed=True)
        if len(simplified) >= 3:
            result.append(simplified)
    return result


def _ocr_region(image: np.ndarray, contour: np.ndarray) -> str:
    """OCR текста внутри контура (геологический код)."""
    try:
        import pytesseract
    except ImportError:
        return ""

    x, y, w, h = cv2.boundingRect(contour)
    if w < 20 or h < 10:
        return ""

    roi = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(
            binary, lang="rus", config="--psm 6"
        ).strip()
    except Exception:
        return ""

    # Ищем геологический код: буква + цифры (C₁, m₃, D₂, ε₁ и т.д.)
    match = re.search(r'[A-Za-zА-Яа-яёЁ][₀-₉\d]*[\^_]?[₀-₉\d]*', text)
    return match.group(0) if match else ""


def _codes_match(ocr_code: str, legend_code: str) -> bool:
    """Нечёткое сравнение геологических кодов (C1 == C₁)."""
    def normalize(s: str) -> str:
        # убираем подстрочные символы Unicode → ASCII цифры
        subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        return s.translate(subs).lower().replace(" ", "")
    return normalize(ocr_code) == normalize(legend_code)
