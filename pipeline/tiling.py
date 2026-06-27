"""
Советская топографическая номенклатура.

Иерархия масштабов:
  1:1 000 000  — лист LETTER-NUMBER  (4°lat × 6°lon)
  1:500 000    — квадрант А/Б/В/Г    (2°lat × 3°lon)
  1:100 000    — лист №1…144         (20'lat × 30'lon)

Функция sheets_to_bbox() парсит имена листов и возвращает объединённый bbox.
Примеры: "M-43-В", "L-42-Б", "M-43-97"
"""

import logging
import math
import re

log = logging.getLogger(__name__)

TILE_LAT = 20 / 60      # 20' в градусах
TILE_LON = 30 / 60      # 30' в градусах
SHEET_LAT = 4.0
SHEET_LON = 6.0
ROWS_100K = 12
COLS_100K = 12

# Квадранты 1:500к (кириллица): А=СЗ, Б=СВ, В=ЮЗ, Г=ЮВ
_QUAD_CYR = {"А": (True, False), "Б": (True, True), "В": (False, False), "Г": (False, True)}
_QUAD_LAT = {"A": (True, False), "B": (True, True), "C": (False, False), "D": (False, True)}


def _parse_sheet_name(name: str) -> tuple[float, float, float, float]:
    """
    Парсит номенклатурное имя листа и возвращает (lat_min, lon_min, lat_max, lon_max).

    Поддерживаемые форматы:
      M-43          — лист 1:1М
      M-43-В        — квадрант 1:500к (кириллица А/Б/В/Г)
      M-43-97       — лист 1:100к (число 1-144)
    """
    name = name.strip().upper()

    # Парсим части: буква-число[-подлист]
    m = re.match(r'^([A-Z])-(\d+)(?:-(.+))?$', name)
    if not m:
        raise ValueError(f"Неверный формат листа: '{name}'")

    letter_str, col_str, sub = m.group(1), m.group(2), m.group(3)

    # 1:1М bbox
    row_idx = ord(letter_str) - ord('A')      # A=0, B=1, …, M=12
    col_num = int(col_str)
    lat_min_1m = row_idx * SHEET_LAT
    lat_max_1m = lat_min_1m + SHEET_LAT
    lon_min_1m = (col_num - 1) * SHEET_LON - 180
    lon_max_1m = lon_min_1m + SHEET_LON

    if sub is None:
        return lat_min_1m, lon_min_1m, lat_max_1m, lon_max_1m

    # Квадрант 1:500к
    if sub in _QUAD_CYR or sub in _QUAD_LAT:
        is_north, is_east = _QUAD_CYR.get(sub) or _QUAD_LAT.get(sub)
        lat_mid = lat_min_1m + SHEET_LAT / 2
        lon_mid = lon_min_1m + SHEET_LON / 2
        lat_min = lat_mid if is_north else lat_min_1m
        lat_max = lat_max_1m if is_north else lat_mid
        lon_min = lon_mid if is_east  else lon_min_1m
        lon_max = lon_max_1m if is_east else lon_mid
        return lat_min, lon_min, lat_max, lon_max

    # Лист 1:100к (число)
    if sub.isdigit():
        n = int(sub)
        if not (1 <= n <= 144):
            raise ValueError(f"Номер листа 1:100к должен быть 1-144, получен {n}")
        row = (n - 1) // COLS_100K   # 0-indexed сверху
        col = (n - 1) % COLS_100K
        lat_max = lat_max_1m - row * TILE_LAT
        lat_min = lat_max - TILE_LAT
        lon_min = lon_min_1m + col * TILE_LON
        lon_max = lon_min + TILE_LON
        return lat_min, lon_min, lat_max, lon_max

    raise ValueError(f"Неизвестный подлист '{sub}' в '{name}'")


def sheets_to_bbox(sheet_names: list[str]) -> tuple[float, float, float, float]:
    """
    Вычисляет объединённый bbox по списку номенклатурных листов.
    Возвращает (lat_min, lon_min, lat_max, lon_max).
    """
    if not sheet_names:
        raise ValueError("Список листов пуст")

    lat_mins, lon_mins, lat_maxs, lon_maxs = [], [], [], []
    for name in sheet_names:
        la0, lo0, la1, lo1 = _parse_sheet_name(name)
        lat_mins.append(la0)
        lon_mins.append(lo0)
        lat_maxs.append(la1)
        lon_maxs.append(lo1)
        log.info(f"    {name}: lat {la0:.4f}–{la1:.4f}, lon {lo0:.4f}–{lo1:.4f}")

    bbox = (min(lat_mins), min(lon_mins), max(lat_maxs), max(lon_maxs))
    log.info(f"    Объединённый bbox: {bbox}")
    return bbox


def _sheet_1m(lat: float, lon: float) -> tuple[str, float, float]:
    letter_idx = int(lat // SHEET_LAT)
    letter = chr(ord('A') + letter_idx)
    zone = int((lon + 180) // SHEET_LON) + 1
    lat_start = letter_idx * SHEET_LAT
    lon_start = (zone - 1) * SHEET_LON - 180
    return f"{letter}-{zone}", lat_start, lon_start


def _sheet_100k_number(lat: float, lon: float, lat_start: float, lon_start: float) -> int:
    lat_end = lat_start + SHEET_LAT
    row = int((lat_end - lat) / TILE_LAT)
    col = int((lon - lon_start) / TILE_LON)
    row = min(max(row, 0), ROWS_100K - 1)
    col = min(max(col, 0), COLS_100K - 1)
    return row * COLS_100K + col + 1


def get_tile_name(lat: float, lon: float) -> str:
    name_1m, lat_start, lon_start = _sheet_1m(lat, lon)
    num = _sheet_100k_number(lat, lon, lat_start, lon_start)
    return f"{name_1m}-{num}"


def generate_tiles(bbox: tuple) -> list[dict]:
    """
    Генерирует все листы 1:100 000 в пределах bbox.
    bbox = (lat_min, lon_min, lat_max, lon_max)
    Возвращает: [{"name": "M-43-97", "bbox": (lat_min, lon_min, lat_max, lon_max)}, ...]
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    seen = set()
    tiles = []

    lat = lat_min + TILE_LAT / 2
    while lat < lat_max:
        lon = lon_min + TILE_LON / 2
        while lon < lon_max:
            tile_lat_min = math.floor(lat / TILE_LAT) * TILE_LAT
            tile_lon_min = math.floor(lon / TILE_LON) * TILE_LON
            name = get_tile_name(lat, lon)
            if name not in seen:
                seen.add(name)
                tiles.append({
                    "name": name,
                    "bbox": (
                        round(tile_lat_min, 6),
                        round(tile_lon_min, 6),
                        round(tile_lat_min + TILE_LAT, 6),
                        round(tile_lon_min + TILE_LON, 6),
                    )
                })
            lon += TILE_LON
        lat += TILE_LAT

    log.info(f"    generate_tiles: {len(tiles)} листов")
    return tiles
