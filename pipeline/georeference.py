import cv2
import numpy as np
from rasterio.transform import from_bounds, xy as rio_xy


def build_transform(image_path: str, bbox: tuple):
    """
    Строит аффинное преобразование пиксель → WGS84.

    bbox = (lat_min, lon_min, lat_max, lon_max)
    Возвращает (cropped_image: np.ndarray, transform: Affine)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

    img = _crop_map_border(img)
    h, w = img.shape[:2]
    lat_min, lon_min, lat_max, lon_max = bbox

    # rasterio from_bounds: west, south, east, north, width, height
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    return img, transform


def pixel_to_coord(row: int, col: int, transform) -> tuple[float, float]:
    """
    Конвертирует пиксельные координаты (row, col) в (lon, lat) WGS84.
    row = y (вниз), col = x (вправо)
    """
    lon, lat = rio_xy(transform, row, col)
    return float(lon), float(lat)


def _crop_map_border(img: np.ndarray) -> np.ndarray:
    """
    Находит внутреннюю рамку карты и обрезает по ней.
    Использует Otsu-порог + поиск прямоугольного контура.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    best_rect = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if not (h * w * 0.30 < area < h * w * 0.95):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) <= 6 and area > best_area:
            best_rect = approx
            best_area = area

    if best_rect is None:
        return img

    x, y, cw, ch = cv2.boundingRect(best_rect)
    return img[y:y + ch, x:x + cw]
