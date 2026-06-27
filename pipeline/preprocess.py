"""
Шаг 3: Предобработка изображения карты.

Убираем шумы, выравниваем контраст.
Работает через OpenCV — не подстраивается под конкретную карту.
"""

import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)


def enhance(image_path: str, output_dir: str, suffix: str = "_enhanced") -> str:
    """
    Загружает изображение, применяет предобработку и сохраняет результат.

    Шаги:
      1. Шумоподавление (fastNlMeansDenoisingColored)
      2. CLAHE на L-канале в LAB-пространстве (равномерный контраст)
      3. Unsharp masking (резкость)

    Возвращает путь к сохранённому файлу.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

    h, w = img.shape[:2]
    log.info(f"    Загружено: {os.path.basename(image_path)} ({w}×{h}px)")

    # 1. Шумоподавление
    log.info("    → Шумоподавление (fastNlMeansDenoisingColored h=7)")
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

    # 2. CLAHE по яркости (LAB-пространство)
    log.info("    → Контраст CLAHE (clipLimit=2.0, tileGrid=8×8)")
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 3. Unsharp masking (выделяем детали)
    log.info("    → Повышение резкости (unsharp mask)")
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Сохраняем рядом с оригиналом
    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1] or ".jpg"
    out_path = os.path.join(output_dir, f"{base}{suffix}{ext}")
    cv2.imwrite(out_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.info(f"    Сохранено: {out_path}")

    return out_path
