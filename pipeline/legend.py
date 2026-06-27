"""
Шаги 5–6: Извлечение легенды и OCR названий фракций.

Стратегия:
  1. PaddleOCR находит ВСЕ текстовые блоки в легенде
  2. Для каждого текстового блока ищем прямоугольник СЛЕВА (= свотч)
  3. Извлекаем цвет из свотча
  4. Spatial join: rect + text → запись легенды
"""

import logging
import os
import re

import cv2
import numpy as np

log = logging.getLogger(__name__)

_HSV_TOL = np.array([12, 60, 60])


def extract_legend(legend_path: str, map_path: str = None,
                   map_rect=None, map_polygon=None,
                   output_dir: str = None, debug_dir: str = None) -> list[dict]:
    same_file = (legend_path == map_path)
    img = cv2.imread(legend_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {legend_path}")

    if same_file:
        log.info("    Легенда встроена — ищем область вне рамки карты")
        legend_img = _detect_legend_region(img, map_rect, map_polygon)
        if legend_img is None or legend_img.size == 0:
            log.warning("    Область легенды не найдена, используем всю карту")
            legend_img = img
    else:
        log.info("    Легенда — отдельный файл")
        # Пропускаем верхние 10% (заголовок)
        h = img.shape[0]
        skip = int(h * 0.10)
        legend_img = img[skip:, :]
        log.info(f"    Пропускаем заголовок: первые {skip}px из {h}px")

    log.info(f"    Область легенды: {legend_img.shape[1]}×{legend_img.shape[0]}px")

    entries = _find_swatches_paddle(legend_img, debug_dir)
    log.info(f"    Swatches найдено: {len(entries)}")

    entries = _deduplicate(entries)

    if output_dir:
        _save_swatches(legend_img, entries, output_dir)
        _save_legend_preview(legend_img, entries, debug_dir or output_dir)

    log.info(f"    Итого записей в легенде: {len(entries)}")
    return entries


# ── PaddleOCR: text-first approach ───────────────────────────────────────────

def _find_swatches_paddle(img: np.ndarray, debug_dir=None) -> list[dict]:
    """
    1. PaddleOCR → все текстовые блоки с координатами
    2. Для каждого текста ищем прямоугольник СЛЕВА (= свотч)
    3. Если нашли — извлекаем цвет, сохраняем пару (свотч, текст)
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        log.warning("    PaddleOCR не установлен — fallback на Tesseract")
        return _find_swatches_tesseract_fallback(img)

    h_img, w_img = img.shape[:2]

    # Масштабируем для OCR (быстрее, Paddle внутри ресайзит сам)
    scale = min(1.0, 2000.0 / max(h_img, w_img))
    ocr_img = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img

    log.info(f"    PaddleOCR ({ocr_img.shape[1]}×{ocr_img.shape[0]}px)...")
    import logging as _logging
    _logging.getLogger("ppocr").setLevel(_logging.ERROR)
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_textline_orientation=False, lang='ru')
    result = ocr.ocr(ocr_img)

    if not result or not result[0]:
        log.warning("    PaddleOCR вернул пустой результат")
        return _find_swatches_tesseract_fallback(img)

    # Обратное масштабирование координат
    inv = 1.0 / scale

    # Извлекаем текстовые блоки из нового формата OCRResult (dict-like)
    ocr_item = result[0]
    rec_texts  = ocr_item.get("rec_texts", [])
    rec_scores = ocr_item.get("rec_scores", [])
    dt_polys   = ocr_item.get("dt_polys", [])
    log.info(f"    PaddleOCR нашёл {len(rec_texts)} текстовых блоков")

    # Предобработка для поиска прямоугольников
    rects = _find_bordered_rects(img)
    log.info(f"    Прямоугольников-кандидатов: {len(rects)}")

    entries = []
    used_rects = set()
    used_text_rows = set()  # не матчим несколько текстов на одну y-полосу

    # Сортируем тексты: сначала те, что левее (ближе к свотчу)
    ocr_lines = list(zip(dt_polys, rec_texts, rec_scores))
    ocr_lines.sort(key=lambda t: min(p[0] for p in t[0]))

    for bbox_ocr, text, score in ocr_lines:
        if score < 0.4:
            continue
        text = text.strip()
        if not text or len(text) < 2:
            continue

        # Координаты текста в исходном масштабе
        tx = int(min(p[0] for p in bbox_ocr) * inv)
        ty = int(min(p[1] for p in bbox_ocr) * inv)
        ty2 = int(max(p[1] for p in bbox_ocr) * inv)
        th = max(ty2 - ty, 10)

        # Не берём второе слово той же строки (y-бин 40px)
        y_bin = ty // 40
        if y_bin in used_text_rows:
            continue

        # Ищем прямоугольник СЛЕВА от текста (только очень близкий: ≤ 80px зазор)
        best_rect = _find_rect_left_of_text(
            rects, tx, ty, th, used_rects,
            max_gap=80, max_left=int(w_img * 0.10)
        )
        if best_rect is None:
            continue

        rx, ry, rw, rh = best_rect
        used_rects.add(best_rect)
        used_text_rows.add(y_bin)

        # Цвет свотча
        hex_color, lower, upper = _extract_swatch_color(img, rx, ry, rw, rh)

        code_m = re.search(r'[A-Za-zА-Яа-яёЁ][A-Za-z0-9]*', text)
        entries.append({
            "name":       text,
            "code":       code_m.group(0) if code_m else "",
            "color_hex":  hex_color,
            "hsv_lower":  lower.tolist(),
            "hsv_upper":  upper.tolist(),
            "swatch_img": img[ry:ry + rh, rx:rx + rw].copy(),
            "swatch_path": None,
            "_bbox":      (rx, ry, rw, rh),
        })

    log.info(f"    Matched rect+text пар: {len(entries)}")
    return entries


def _find_bordered_rects(img: np.ndarray) -> list[tuple]:
    """
    Находит bordered rectangles в изображении.
    Сначала убираем длинные линии таблицы, потом ищем свотчи.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Убираем длинные горизонтальные линии (таблица)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 8, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    # Убираем длинные вертикальные линии
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 8))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    long_lines = cv2.bitwise_or(h_lines, v_lines)
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    long_lines = cv2.dilate(long_lines, dil_k, iterations=1)
    clean = cv2.bitwise_and(binary, cv2.bitwise_not(long_lines))

    # Закрываем разрывы в рамках свотчей
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.dilate(clean, k, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_w = max(40, w_img * 0.006)   # свотчи крупнее символов текста
    max_w = max(250, w_img * 0.08)
    min_h = max(30, h_img * 0.005)
    max_h = max(200, h_img * 0.07)

    seen = set()
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (min_w <= cw <= max_w and min_h <= ch <= max_h):
            continue
        if cw / max(ch, 1) > 6 or ch / max(cw, 1) > 4:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.08 * peri, True)
        if not (4 <= len(approx) <= 20):
            continue
        if cv2.contourArea(c) < cw * ch * 0.03:
            continue
        # Снотч должен содержать светлые пиксели (не сплошной текст)
        roi_g = gray[y:y + ch, x:x + cw]
        light_pct = float((roi_g > 100).sum()) / max(roi_g.size, 1)
        if light_pct < 0.20:
            continue
        key = (x // 10, y // 10)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((x, y, cw, ch))

    return sorted(candidates, key=lambda c: (c[1], c[0]))


def _find_rect_left_of_text(rects, tx, ty, th,
                             used_rects, max_gap=400, max_left=600) -> tuple | None:
    """
    Среди найденных прямоугольников ищем тот, который:
    - заканчивается ЛЕВЕЕ начала текста (с разрывом <= max_gap)
    - вертикально перекрывается с текстом
    - не использован другим текстом
    """
    cy = ty + th / 2
    best = None
    best_score = float('inf')  # чем меньше gap + vertical_offset, тем лучше

    for rect in rects:
        if rect in used_rects:
            continue
        rx, ry, rw, rh = rect
        # Прямоугольник должен заканчиваться левее текста
        rect_right = rx + rw
        gap = tx - rect_right
        if gap < -rw * 0.5 or gap > max_gap:  # не накладывается и не слишком далеко
            continue
        # Горизонтальный диапазон: прямоугольник не должен быть слишком далеко влево
        if tx - rx > max_left:
            continue
        # Вертикальное перекрытие
        rect_cy = ry + rh / 2
        v_dist = abs(rect_cy - cy)
        if v_dist > th * 2.0:
            continue
        score = gap * 0.7 + v_dist * 0.3
        if score < best_score:
            best_score = score
            best = rect

    return best


def _extract_swatch_color(img, rx, ry, rw, rh):
    """Извлекает медианный цвет из interior свотча."""
    pad = max(2, min(rh // 6, rw // 6))
    roi = img[ry + pad:ry + rh - pad, rx + pad:rx + rw - pad]
    if roi.size == 0:
        roi = img[ry:ry + rh, rx:rx + rw]

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mask = roi_lab[:, :, 0] > 40
    if mask.sum() < 4:
        mask = np.ones_like(mask, dtype=bool)

    flat = roi.reshape(-1, 3)[mask.ravel()]
    if flat.size == 0:
        flat = roi.reshape(-1, 3)
    median_bgr = np.median(flat, axis=0).astype(int)

    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])
    )
    bgr_px = np.array([[[int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2])]]], dtype=np.uint8)
    hsv_px = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    lower = np.clip(hsv_px - _HSV_TOL, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    upper = np.clip(hsv_px + _HSV_TOL, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

    return hex_color, lower, upper


# ── Tesseract fallback ────────────────────────────────────────────────────────

def _find_swatches_tesseract_fallback(img: np.ndarray) -> list[dict]:
    """Старый подход: Canny→прямоугольники, Tesseract OCR справа."""
    rects = _find_bordered_rects(img)
    if not rects:
        return []

    entries = []
    for (rx, ry, rw, rh) in rects:
        hex_color, lower, upper = _extract_swatch_color(img, rx, ry, rw, rh)
        code_m = re.search(r'[A-Za-zА-Яа-яёЁ]', hex_color)
        entries.append({
            "name": hex_color, "code": "",
            "color_hex": hex_color,
            "hsv_lower": lower.tolist(), "hsv_upper": upper.tolist(),
            "swatch_img": img[ry:ry + rh, rx:rx + rw].copy(),
            "swatch_path": None, "_bbox": (rx, ry, rw, rh),
        })

    _ocr_labels_tesseract(img, entries)
    return entries


def _ocr_labels_tesseract(img: np.ndarray, entries: list[dict]) -> None:
    try:
        import pytesseract
        from PIL import Image as PILImage
    except ImportError:
        return

    h_img, w_img = img.shape[:2]
    for entry in entries:
        x, y, cw, ch = entry["_bbox"]
        tx = x + cw + 4
        tw = min(600, w_img - tx)
        if tw <= 0:
            continue
        roi = img[max(0, y - 4):y + ch + 8, tx:tx + tw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        scale = max(1, int(60 / max(ch, 1)) + 1)
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            text = pytesseract.image_to_string(PILImage.fromarray(binary),
                                               config="--psm 7 -l rus+eng").strip()
        except Exception:
            continue
        if text:
            code_m = re.search(r'[A-Za-zА-Яа-яёЁ][A-Za-z0-9]*', text)
            entry["code"] = code_m.group(0) if code_m else ""
            entry["name"] = text


# ── Детекция области легенды (embedded case) ──────────────────────────────────

def _detect_legend_region(img: np.ndarray, map_rect=None,
                           map_polygon=None) -> np.ndarray | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    outside = np.ones((h, w), dtype=np.uint8) * 255
    if map_polygon is not None and len(map_polygon) >= 3:
        pts = np.array(map_polygon, dtype=np.int32)
        cv2.fillPoly(outside, [pts], 0)
    elif map_rect is not None:
        mx, my, mw, mh = map_rect
        outside[my:my + mh, mx:mx + mw] = 0

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    outside_only = cv2.bitwise_and(closed, outside)

    contours, _ = cv2.findContours(outside_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lo = h * w * 0.00001
    hi = h * w * 0.002
    pts_list = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (lo < area < hi):
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        if max(rw, rh) / max(min(rw, rh), 1) > 8:
            continue
        pts_list.append((rx + rw // 2, ry + rh // 2))

    log.info(f"    Swatch-объектов вне рамки: {len(pts_list)}")

    if len(pts_list) < 3:
        if map_rect is not None:
            mx, my, mw, mh = map_rect
            below = img[my + mh:, mx:mx + mw]
            if below.shape[0] > 10:
                return below
        return None

    xs = [p[0] for p in pts_list]
    ys = [p[1] for p in pts_list]
    pad_x = max(int(w * 0.05), int((max(xs) - min(xs)) * 0.3))
    pad_y = max(int(h * 0.02), int((max(ys) - min(ys)) * 0.3))
    lx1 = max(0, min(xs) - pad_x)
    ly1 = max(0, min(ys) - pad_y)
    lx2 = min(w, max(xs) + pad_x)
    ly2 = min(h, max(ys) + pad_y)
    return img[ly1:ly2, lx1:lx2]


# ── Дедупликация ──────────────────────────────────────────────────────────────

def _deduplicate(entries: list[dict]) -> list[dict]:
    seen_bboxes = set()
    unique = []
    for entry in entries:
        bbox = entry.get("_bbox")
        key = bbox if bbox else entry["color_hex"]
        if key not in seen_bboxes:
            seen_bboxes.add(key)
            unique.append(entry)
    return unique


# ── Сохранение ────────────────────────────────────────────────────────────────

def _save_swatches(legend_img: np.ndarray, entries: list[dict],
                   output_dir: str) -> None:
    swatch_dir = os.path.join(output_dir, "swatches")
    os.makedirs(swatch_dir, exist_ok=True)
    for i, entry in enumerate(entries):
        swatch = entry.get("swatch_img")
        if swatch is None or swatch.size == 0:
            continue
        code = re.sub(r'[^\w]', '_', entry.get("code", "")) or f"color_{i}"
        path = os.path.join(swatch_dir, f"{i:03d}_{code}.jpg")
        cv2.imwrite(path, swatch)
        entry["swatch_path"] = path
    log.info(f"    Swatches сохранены → {swatch_dir}/")


def _save_legend_preview(legend_img: np.ndarray, entries: list[dict],
                          output_dir: str) -> None:
    vis = legend_img.copy()
    for entry in entries:
        bbox = entry.get("_bbox")
        if bbox:
            x, y, cw, ch = bbox
            cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 220, 0), 2)
    path = os.path.join(output_dir, "legend_preview.jpg")
    h, w = vis.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    cv2.imwrite(path, cv2.resize(vis, (int(w * scale), int(h * scale))))
    log.info(f"    Превью легенды → {path}")
