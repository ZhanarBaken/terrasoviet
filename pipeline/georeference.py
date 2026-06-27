"""
Шаг 4: Детекция границы карты и геопривязка.

Алгоритм поиска границы (универсальный, два метода):

  Метод А — яркостной профиль (primary):
    Граница карты находится там, где яркость резко падает от светлого поля
    (белое поле бумаги) к тёмному содержимому карты.
    Сканируем строки/столбцы от каждого края внутрь, ищем первый порог.

  Метод Б — морфологический (fallback для карт без белого поля):
    Canny-рёбра → дилатация (заполнение пробелов) → MORPH_OPEN с длинным
    ядром (≥20% стороны). Выживают только непрерывные прямые линии нужной длины.
    Берём КРАЙНИЕ строки/столбцы — это и есть рамка.

Граница может иметь 4+ углов (Soviet maps с нестандартной рамкой).

Возвращает: (cropped_img, map_rect, debug_vis, polygon)
"""

import logging
import os

import cv2
import numpy as np
from rasterio.transform import from_bounds, xy as rio_xy

log = logging.getLogger(__name__)


def build_transform(image_path: str, bbox: tuple, output_dir: str = None):
    """
    Загружает изображение, находит рамку карты, строит аффинную геопривязку.

    Возвращает: (cropped_img, rasterio_transform, map_rect, polygon)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

    log.info(f"    Изображение: {img.shape[1]}×{img.shape[0]}px")

    cropped, map_rect, vis, polygon = find_map_border(img)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, "border_detection.jpg")
        h_v, w_v = vis.shape[:2]
        scale = min(1.0, 1200 / max(h_v, w_v))
        cv2.imwrite(vis_path,
                    cv2.resize(vis, (int(w_v * scale), int(h_v * scale))),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.info(f"    Визуализация рамки → {vis_path}")

    h, w = cropped.shape[:2]
    lat_min, lon_min, lat_max, lon_max = bbox
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    return cropped, transform, map_rect, polygon


def pixel_to_coord(row: int, col: int, transform) -> tuple[float, float]:
    lon, lat = rio_xy(transform, row, col)
    return float(lon), float(lat)


def find_map_border(img: np.ndarray):
    """
    Находит рамку карты — многоугольник из длинных прямых линий.

    Пробует два метода:
      1. Яркостной профиль (row/col mean) → переход от светлого поля к карте
      2. Морф. фильтрация Canny-рёбер (если нет чёткого светлого поля)

    Рамка может иметь 4+ углов.

    Возвращает: (cropped_img, (x, y, w, h), debug_vis, polygon)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    log.info(f"    Детекция рамки ({w}×{h}px)")

    # ── Метод А: Hough — длинные прямые линии (универсальный) ────────────
    y_top, y_bot, x_lft, x_rgt = _border_by_hough(gray, w, h)

    if y_top is not None:
        log.info("    Метод А (Hough линии):")
        log.info(f"      top={y_top} bot={y_bot} lft={x_lft} rgt={x_rgt}")
        h_pos = [y_top, y_bot]
        v_pos = [x_lft, x_rgt]
    else:
        # ── Метод Б: яркостной профиль ────────────────────────────────────
        y_top, y_bot, x_lft, x_rgt = _border_by_brightness(gray)
        if y_top is not None:
            log.info("    Метод Б (яркостной профиль):")
            log.info(f"      top={y_top} bot={y_bot} lft={x_lft} rgt={x_rgt}")
            h_pos = [y_top, y_bot]
            v_pos = [x_lft, x_rgt]
        else:
            # ── Метод В: морфологический ──────────────────────────────────
            log.info("    Метод В (морфологический):")
            h_pos, v_pos = _border_by_morphology(gray, w, h)
            log.info(f"      H линий: {h_pos}  V линий: {v_pos}")

    # ── Строим polygon из всех пересечений H×V ─────────────────────────
    corners = [(vx, hy) for hy in h_pos for vx in v_pos]
    if len(corners) >= 4:
        pts = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        polygon = [tuple(int(v) for v in p[0]) for p in hull]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_lft_f, x_rgt_f = min(xs), max(xs)
        y_top_f, y_bot_f = min(ys), max(ys)
    else:
        log.warning("    Недостаточно угловых точек — берём крайние позиции")
        y_top_f = h_pos[0] if h_pos else 0
        y_bot_f = h_pos[-1] if h_pos else h
        x_lft_f = v_pos[0] if v_pos else 0
        x_rgt_f = v_pos[-1] if v_pos else w
        polygon = [(x_lft_f, y_top_f), (x_rgt_f, y_top_f),
                   (x_rgt_f, y_bot_f), (x_lft_f, y_bot_f)]

    map_rect = (x_lft_f, y_top_f, x_rgt_f - x_lft_f, y_bot_f - y_top_f)
    log.info(f"    Рамка: x={x_lft_f} y={y_top_f} "
             f"{x_rgt_f - x_lft_f}×{y_bot_f - y_top_f}px, углов={len(polygon)}")

    # ── Визуализация ───────────────────────────────────────────────────────
    vis = img.copy()
    thick = max(3, min(w, h) // 400)
    pts_draw = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts_draw], True, (0, 220, 0), thick * 2)

    cropped = img[y_top_f:y_bot_f, x_lft_f:x_rgt_f]
    if cropped.size == 0:
        log.warning("    Кроп пустой — возвращаю оригинал")
        return img, (0, 0, w, h), vis, [(0, 0), (w, 0), (w, h), (0, h)]

    return cropped, map_rect, vis, polygon


# ── Методы детекции ────────────────────────────────────────────────────────

def _border_by_hough(gray: np.ndarray, w: int, h: int):
    """
    Ищет рамку карты через HoughLinesP.

    Рамка карты = самые длинные горизонтальные и вертикальные линии.
    Работает независимо от расположения карты в изображении:
    не важно есть ли белое поле, где именно рамка находится.

    Критерий линии:
      - горизонтальная: dy/dx < tan(3°) ≈ 0.05
      - вертикальная:   dx/dy < tan(3°) ≈ 0.05
      - длина ≥ 25% меньшей стороны изображения
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    min_len = int(min(w, h) * 0.25)
    max_gap = int(min(w, h) * 0.015)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    if lines is None:
        return None, None, None, None

    h_lines = []  # (y_mid, segment_length)
    v_lines = []  # (x_mid, segment_length)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = float(np.hypot(dx, dy))

        if dy == 0 or (dx > 0 and dy / dx < 0.05):      # горизонтальная
            h_lines.append(((y1 + y2) / 2, length))
        elif dx == 0 or (dy > 0 and dx / dy < 0.05):    # вертикальная
            v_lines.append(((x1 + x2) / 2, length))

    # Отбираем только достаточно длинные
    h_long = sorted([(y, l) for y, l in h_lines if l >= w * 0.3], key=lambda t: t[0])
    v_long = sorted([(x, l) for x, l in v_lines if l >= h * 0.3], key=lambda t: t[0])

    if len(h_long) < 2 or len(v_long) < 2:
        log.info(f"      Hough: H-линий={len(h_long)} V-линий={len(v_long)} — мало")
        return None, None, None, None

    y_top = int(h_long[0][0])
    y_bot = int(h_long[-1][0])
    x_lft = int(v_long[0][0])
    x_rgt = int(v_long[-1][0])

    if (y_bot - y_top) < h * 0.25 or (x_rgt - x_lft) < w * 0.25:
        log.info(f"      Hough: рамка слишком маленькая — пропускаем")
        return None, None, None, None

    return y_top, y_bot, x_lft, x_rgt


def _border_by_brightness(gray: np.ndarray):
    """
    Ищет границу через профиль яркости строк/столбцов.

    Принцип: белое поле (margin) имеет среднюю яркость ≥218.
    Рамка = первая строка/столбец где среднее падает на ≥12 ниже поля.

    Возвращает (y_top, y_bot, x_lft, x_rgt) или (None,None,None,None) если
    нет чёткого светлого поля.
    """
    h, w = gray.shape

    row_mean = np.mean(gray, axis=1)
    col_mean = np.mean(gray, axis=0)

    ref_len = max(5, int(h * 0.05))

    def scan_forward(profile, ref_len, drop=12):
        ref = float(np.mean(profile[:ref_len]))
        if ref < 218:
            return None
        thr = ref - drop
        for i, v in enumerate(profile):
            if float(v) < thr:
                return i
        return None

    def scan_backward(profile, ref_len, drop=12):
        ref = float(np.mean(profile[-ref_len:]))
        if ref < 218:
            return None
        thr = ref - drop
        n = len(profile)
        for i in range(n - 1, -1, -1):
            if float(profile[i]) < thr:
                return i
        return None

    y_top = scan_forward(row_mean, ref_len)
    y_bot = scan_backward(row_mean, ref_len)
    x_lft = scan_forward(col_mean, max(5, int(w * 0.05)))
    x_rgt = scan_backward(col_mean, max(5, int(w * 0.05)))

    if None in (y_top, y_bot, x_lft, x_rgt):
        return None, None, None, None

    # Проверяем адекватность: рамка должна занимать ≥30% стороны
    if (y_bot - y_top) < h * 0.30 or (x_rgt - x_lft) < w * 0.30:
        log.warning("    Яркостной метод: слишком маленькая рамка, пропускаем")
        return None, None, None, None

    return int(y_top), int(y_bot), int(x_lft), int(x_rgt)


def _border_by_morphology(gray: np.ndarray, w: int, h: int):
    """
    Ищет границу через Canny → gap-fill → MORPH_OPEN.

    Возвращает (h_pos_list, v_pos_list) — y-позиции горизонтальных
    и x-позиции вертикальных граничных линий.
    """
    edges = cv2.Canny(gray, 30, 90)

    gap = max(3, int(min(w, h) * 0.01))   # 1% стороны
    kw = max(w // 5, 50)                   # фильтр: ≥20% ширины
    kh = max(h // 5, 50)                   # фильтр: ≥20% высоты

    e_h = cv2.dilate(edges, np.ones((1, gap), np.uint8))
    h_mask = cv2.morphologyEx(e_h, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))

    e_v = cv2.dilate(edges, np.ones((gap, 1), np.uint8))
    v_mask = cv2.morphologyEx(e_v, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))

    h_rows = np.where(np.any(h_mask, axis=1))[0]
    v_cols = np.where(np.any(v_mask, axis=0))[0]

    log.info(f"      H-строк: {len(h_rows)}  V-столбцов: {len(v_cols)}")

    def edge_positions(indices, dim):
        """
        Берём ПЕРВЫЙ и ПОСЛЕДНИЙ индекс — это крайние линии рамки.
        Любые геологические линии между ними — внутри карты, не рамка.
        """
        if len(indices) == 0:
            return [0, dim]
        first = int(indices[0])
        last  = int(indices[-1])
        if abs(last - first) < dim * 0.10:   # одна и та же линия
            return [first]
        return [first, last]

    h_pos = edge_positions(h_rows, h)
    v_pos = edge_positions(v_cols, w)

    return h_pos, v_pos
