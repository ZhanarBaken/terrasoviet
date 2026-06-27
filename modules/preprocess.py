import cv2
import numpy as np


def preprocess(image_path: str) -> np.ndarray:
    """
    Вход:  путь к файлу (str)
    Выход: numpy array (H, W, 3), BGR, uint8
           уже обрезанный и денойзнутый

    Шаги:
        1. Загрузка
        2. Обрезка рамки карты
        3. Denoising
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

    img = _crop_border(img)
    img = _denoise(img)
    return img


def classify_map(image: np.ndarray) -> str:
    """
    Вход:  numpy array после preprocess()
    Выход: "color" | "bw" | "composite"

    Логика: по среднему насыщению HSV
        < 20  → "bw"
        < 50  → "composite"
        >= 50 → "color"
    Пороги подобрать на реальных картах!
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = hsv[:, :, 1].mean()

    if mean_saturation < 20:
        return "bw"
    elif mean_saturation < 50:
        return "composite"
    else:
        return "color"


def _crop_border(img: np.ndarray) -> np.ndarray:
    """
    TODO: обрезать внешнюю рамку карты
    Идея: найти прямоугольный контур рамки через
          cv2.findContours или Hough, обрезать по нему
    """
    # Заглушка — возвращаем как есть
    return img


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    TODO: убрать шум сканирования
    Порядок:
        1. cv2.medianBlur — крупные артефакты
        2. cv2.fastNlMeansDenoisingColored — мелкий шум
        (morphologyEx применяется позже, после сегментации)
    """
    # Заглушка — возвращаем как есть
    return img
