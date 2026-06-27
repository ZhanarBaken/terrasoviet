"""
Извлечение цветовой палитры и названий формаций из легенды.

Стратегия:
  1. Найти цветные прямоугольники (swatches) в legenda.jpg
  2. Взять медианный HSV-цвет каждого swatch
  3. OCR текста справа (геологический код + название)
  4. Вернуть список записей для segment.py

Классификация в segment.py использует ОБА критерия:
  - цвет (HSV inRange)
  - текст внутри региона (OCR код вроде C₁, m₃)
"""

import logging
import re

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Допуск HSV при поиске цвета на карте
_HSV_TOL = np.array([12, 60, 60])


def extract_legend(legend_path: str, map_path: str = None) -> list[dict]:
    """
    Вход:  путь к легенде (отдельный файл ИЛИ совпадает с картой),
           опционально путь к карте для fallback KMeans.
    Выход: [{"name": str, "code": str, "color_hex": str,
              "hsv_lower": list, "hsv_upper": list}, ...]

    Стратегия:
      1. Если legend_path — отдельный файл: читаем его напрямую
      2. Если legend_path == map_path (легенда внутри карты):
         автоматически вырезаем блок легенды из углов карты
      3. Если нашли мало swatches (<5): KMeans fallback по карте
    """
    # Определяем источник легенды
    same_file = (legend_path == map_path)

    img = cv2.imread(legend_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {legend_path}")

    if same_file:
        log.info("Легенда встроена в карту — ищу блок легенды автоматически")
        legend_img = _detect_embedded_legend(img)
        if legend_img is None:
            log.warning("Блок легенды не найден, использую всю карту")
            legend_img = img
    else:
        legend_img = img

    entries = _find_swatches(legend_img)
    entries = _add_ocr_labels(legend_img, entries)
    entries = _deduplicate(entries)

    if len(entries) < 5 and map_path:
        log.warning(f"Легенда: найдено мало записей ({len(entries)}), "
                    "переключаюсь на KMeans по карте")
        entries = _kmeans_from_map(map_path)

    log.info(f"Легенда: итого {len(entries)} записей")
    return entries


def _detect_embedded_legend(img: np.ndarray) -> np.ndarray | None:
    """
    Находит блок легенды внутри карты.

    Алгоритм (аналогично поиску внешней рамки):
      1. Ищем длинные горизонтальные + вертикальные линии (структура таблицы/рамки)
      2. Объединяем в блоки через дилатацию
      3. Из блоков в углах карты выбираем самый светлый с цветными swatches
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Находим длинные линии (как в таблице легенды)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 15))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_h),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_v),
    )
    # Дилатируем чтобы замкнуть прямоугольники
    dilated = cv2.dilate(lines, np.ones((30, 30), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # Легенда: 0.1–20% площади карты
        if not (h * w * 0.001 < area < h * w * 0.20):
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        # Должна быть в одном из 4 углов (40% от края)
        in_corner = (
            (x < w * 0.40 or x + cw > w * 0.60) and
            (y < h * 0.40 or y + ch > h * 0.60)
        )
        if not in_corner:
            continue

        roi = img[y:y + ch, x:x + cw]

        # Светлый фон — легенда это бумага, не карта
        brightness = cv2.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))[0]
        if brightness < 150:
            continue

        # Число цветных swatches внутри
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_roi, (0, 40, 80), (179, 255, 255))
        sw_cnts, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        swatch_count = sum(
            1 for sc in sw_cnts
            if cw * ch * 0.00005 < cv2.contourArea(sc) < cw * ch * 0.05
        )

        candidates.append((swatch_count, area, roi, x, y, cw, ch, brightness))
        log.info(f"  Кандидат легенды: ({x},{y}) {cw}×{ch}px "
                 f"bright={brightness:.0f} swatches={swatch_count}")

    if not candidates:
        return None

    # Лучший = больше swatches, при равенстве — крупнее
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = candidates[0]
    log.info(f"Легенда найдена в карте: ({best[3]},{best[4]}) "
             f"{best[5]}×{best[6]}px swatches={best[0]}")
    return best[2]


def _find_swatches(img: np.ndarray) -> list[dict]:
    """Находит цветные прямоугольники-образцы в изображении легенды."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]

    # Порог S>80 чтобы отсечь жёлтую бумагу (S~40-60)
    mask = cv2.inRange(hsv, (0, 80, 80), (179, 255, 240))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    entries = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        # Фильтр по размеру: swatch — небольшой прямоугольник
        if cw < 12 or ch < 8:
            continue
        if cw > w_img * 0.20 or ch > h_img * 0.05:
            continue
        if cw / ch > 6 or ch / cw > 4:
            continue

        # Медианный HSV цвет swatch (исключаем краевые пиксели)
        pad = 2
        roi = hsv[y + pad:y + ch - pad, x + pad:x + cw - pad]
        if roi.size == 0:
            continue
        median_hsv = np.median(roi.reshape(-1, 3), axis=0)

        lower = np.clip(median_hsv - _HSV_TOL, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        upper = np.clip(median_hsv + _HSV_TOL, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

        bgr = img[y + ch // 2, x + cw // 2].tolist()
        hex_color = "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])

        entries.append({
            "name": hex_color,      # будет перезаписан OCR
            "code": "",             # геологический код (C₁, m₃, ...)
            "color_hex": hex_color,
            "hsv_lower": lower.tolist(),
            "hsv_upper": upper.tolist(),
            "_bbox": (x, y, cw, ch),
        })

    # Сортируем сверху-вниз, слева-направо (порядок легенды)
    entries.sort(key=lambda e: (e["_bbox"][1], e["_bbox"][0]))
    return entries


def _add_ocr_labels(img: np.ndarray, entries: list[dict]) -> list[dict]:
    """OCR текста справа от каждого swatch (EasyOCR)."""
    try:
        import easyocr
        reader = easyocr.Reader(["en", "ru"], gpu=False, verbose=False)
    except Exception:
        log.warning("easyocr недоступен — OCR пропущен, используем hex-цвета как имена")
        return entries

    h_img, w_img = img.shape[:2]

    for entry in entries:
        x, y, cw, ch = entry["_bbox"]
        tx = x + cw + 4
        tw = min(500, w_img - tx)
        ty = max(0, y - 2)
        th = ch + 4
        if tw <= 0 or tx >= w_img:
            continue

        roi = img[ty:ty + th, tx:tx + tw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            results = reader.readtext(binary, detail=0, paragraph=True)
            text = " ".join(results).strip()
        except Exception:
            continue

        if not text:
            continue

        code_match = re.search(r'[A-Za-zА-Яа-яёЁ]\d*[\^_]?\d*', text)
        entry["code"] = code_match.group(0) if code_match else ""
        entry["name"] = text

    return entries


def _deduplicate(entries: list[dict]) -> list[dict]:
    """Убирает записи с почти одинаковыми HSV-цветами."""
    unique = []
    for entry in entries:
        center = np.array(entry["hsv_lower"], dtype=float) + np.array(entry["hsv_upper"], dtype=float)
        is_dup = False
        for u in unique:
            u_center = np.array(u["hsv_lower"], dtype=float) + np.array(u["hsv_upper"], dtype=float)
            if np.linalg.norm(center - u_center) < 25:
                is_dup = True
                break
        if not is_dup:
            unique.append(entry)

    # Убираем служебное поле
    for e in unique:
        e.pop("_bbox", None)

    return unique


def _kmeans_from_map(map_path: str, n_colors: int = 30) -> list[dict]:
    """
    Fallback: находит n доминирующих цветов прямо на карте через KMeans.
    Фильтрует фоновые (малонасыщенные) цвета.
    """
    from sklearn.cluster import KMeans

    img = cv2.imread(map_path)
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape(-1, 3).astype(float)
    # Сэмплируем для скорости
    idx = np.random.choice(len(pixels), min(80000, len(pixels)), replace=False)
    sample = pixels[idx]

    km = KMeans(n_clusters=n_colors, random_state=42, n_init=5)
    km.fit(sample)

    entries = []
    for center in km.cluster_centers_:
        h, s, v = center
        # Пропускаем малонасыщенные (фон бумаги)
        if s < 45 or v < 60:
            continue

        median_hsv = np.array([h, s, v])
        lower = np.clip(median_hsv - _HSV_TOL, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        upper = np.clip(median_hsv + _HSV_TOL, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

        bgr = cv2.cvtColor(np.uint8([[[int(h), int(s), int(v)]]]), cv2.COLOR_HSV2BGR)[0][0]
        hex_color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

        entries.append({
            "name": hex_color,
            "code": "",
            "color_hex": hex_color,
            "hsv_lower": lower.tolist(),
            "hsv_upper": upper.tolist(),
        })

    log.info(f"KMeans: {len(entries)} цветных кластеров из {n_colors}")
    return entries
