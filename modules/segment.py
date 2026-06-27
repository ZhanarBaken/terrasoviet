import cv2
import numpy as np


def segment(image: np.ndarray, map_type: str) -> dict:
    """
    Вход:
        image    — numpy array (H, W, 3), BGR, после preprocess()
        map_type — "color" | "bw" | "composite"
    Выход:
        {
            "layers": [{"name": str, "mask": np.ndarray (H,W) uint8}],
            "faults": [{"contour": np.ndarray (N,1,2) int32}]
        }

    Логика по типу:
        "color"     → HSV сегментация по цвету слоёв
        "bw"        → яркостной порог + контуры
        "composite" → разбить на субкарты, каждую обработать отдельно
    """
    if map_type == "color":
        return _segment_color(image)
    elif map_type == "bw":
        return _segment_bw(image)
    elif map_type == "composite":
        return _segment_composite(image)
    else:
        raise ValueError(f"Неизвестный тип карты: {map_type}")


def _segment_color(image: np.ndarray) -> dict:
    """
    TODO: HSV сегментация цветных карт (как 35.jpg, 21.jpg)
    Цвета для поиска (подобрать диапазоны по реальным картам):
        синий/фиолетовый → разломы/жилы
        красный/оранжевый → особые объекты
        жёлтый → геологические тела
        розовый → другие слои
    Пример:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, (100,50,50), (140,255,255))
    """
    return {"layers": [], "faults": []}


def _segment_bw(image: np.ndarray) -> dict:
    """
    TODO: обработка ч/б карт (как 31.jpg)
    Идея:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, ...)
        contours = cv2.findContours(...)
    """
    return {"layers": [], "faults": []}


def _segment_composite(image: np.ndarray) -> dict:
    """
    TODO: составные карты (как 46.jpg) — несколько субкарт на листе
    Идея:
        1. Найти прямоугольные рамки субкарт через findContours
        2. Вырезать каждую субкарту
        3. Прогнать через _segment_color или _segment_bw
        4. Объединить результаты
    """
    return {"layers": [], "faults": []}
