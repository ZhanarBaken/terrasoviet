"""
Советская топографическая номенклатура 1:100 000.

Лист 1:1М  (4° lat × 6° lon) → 144 листа 1:100к (12×12 сетка, 20'×30' каждый)
Нумерация: слева-направо, сверху-вниз (1–144).

Квадранты 1:500к внутри листа 1:1М:
    А = СЗ (top-left),  Б = СВ (top-right)
    В = ЮЗ (bottom-left), Г = ЮВ (bottom-right)
"""

import math

TILE_LAT = 20 / 60        # 20 минут в градусах
TILE_LON = 30 / 60        # 30 минут в градусах
SHEET_LAT = 4.0
SHEET_LON = 6.0
ROWS = 12
COLS = 12

_QUADRANT = {
    (True, False): 'А',   # сев. полов. + зап. полов.
    (True, True):  'Б',   # сев. полов. + вост. полов.
    (False, False): 'В',  # юж. полов. + зап. полов.
    (False, True):  'Г',  # юж. полов. + вост. полов.
}


def _sheet_1m(lat: float, lon: float) -> tuple[str, float, float]:
    """Возвращает (имя листа 1:1М, lat_start, lon_start)."""
    letter_idx = int(lat // SHEET_LAT)
    letter = chr(ord('A') + letter_idx)
    zone = int((lon + 180) // SHEET_LON) + 1
    lat_start = letter_idx * SHEET_LAT
    lon_start = (zone - 1) * SHEET_LON - 180
    return f"{letter}-{zone}", lat_start, lon_start


def _sheet_500k_quadrant(lat: float, lon: float, lat_start: float, lon_start: float) -> str:
    lat_mid = lat_start + SHEET_LAT / 2
    lon_mid = lon_start + SHEET_LON / 2
    return _QUADRANT[(lat >= lat_mid, lon >= lon_mid)]


def _sheet_100k_number(lat: float, lon: float, lat_start: float, lon_start: float) -> int:
    lat_end = lat_start + SHEET_LAT
    row = int((lat_end - lat) / TILE_LAT)
    col = int((lon - lon_start) / TILE_LON)
    row = min(max(row, 0), ROWS - 1)
    col = min(max(col, 0), COLS - 1)
    return row * COLS + col + 1


def get_tile_name(lat: float, lon: float) -> str:
    """Возвращает советский номенклатурный номер листа 1:100 000."""
    name_1m, lat_start, lon_start = _sheet_1m(lat, lon)
    num = _sheet_100k_number(lat, lon, lat_start, lon_start)
    return f"{name_1m}-{num}"


def generate_tiles(bbox: tuple) -> list[dict]:
    """
    Генерирует все листы 1:100 000 в пределах bbox.

    bbox = (lat_min, lon_min, lat_max, lon_max)
    Возвращает список:
        [{"name": "M-43-97", "bbox": (lat_min, lon_min, lat_max, lon_max)}, ...]
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

    return tiles
