import cv2
import numpy as np


def preprocess(image_path: str) -> np.ndarray:
    """
    Вход:  путь к файлу (str)
    Выход: numpy array (H, W, 3), BGR, uint8
           уже обрезанный и денойзнутый
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

    Стратегия:
      1. Мало насыщенных пикселей → "bw"
      2. Несколько прямоугольных панелей на листе → "composite"
      3. Иначе → "color"
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    # доля пикселей с ощутимым цветом (S > 30)
    color_ratio = (s > 30).mean()

    if color_ratio < 0.05:
        return "bw"

    if _has_multiple_panels(image):
        return "composite"

    return "color"


def _has_multiple_panels(img: np.ndarray) -> bool:
    """
    Ищет несколько крупных прямоугольных рамок — признак составной карты.
    Консервативно: возвращает True только при явных 2+ панелях.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu автоматически находит порог под конкретное изображение
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # RETR_LIST — все контуры включая вложенные (субкарты внутри страницы)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_panel = h * w * 0.04
    max_panel = h * w * 0.75  # строже: не весь лист

    panels = 0
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_panel < area < max_panel):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) <= 6:
            x, y, cw, ch = cv2.boundingRect(approx)
            aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
            if aspect < 5:
                panels += 1

    return panels >= 2


def _crop_border(img: np.ndarray) -> np.ndarray:
    """
    Находит основную прямоугольную рамку карты и обрезает по ней.
    Fallback: возвращает исходник если рамка не найдена.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu вместо хардкод порога — работает на картах разного возраста и яркости
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    min_area = h * w * 0.30
    max_area = h * w * 0.95

    best_rect = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # допускаем до 6 вершин — скос/замятость углов при сканировании
        if len(approx) <= 6 and area > best_area:
            best_rect = approx
            best_area = area

    if best_rect is None:
        all_pts = np.vstack(contours)
        x, y, cw, ch = cv2.boundingRect(all_pts)
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        return img[y1:y2, x1:x2]

    x, y, cw, ch = cv2.boundingRect(best_rect)
    return img[y:y + ch, x:x + cw]


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    medianBlur убирает точечные артефакты сканирования,
    bilateralFilter сглаживает зернистость сохраняя края линий.
    В 20x быстрее fastNlMeansDenoisingColored при сопоставимом качестве.
    """
    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img
