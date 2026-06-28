"""
TerraSoviet — векторизация советских геологических карт.

Вход:
  --map     путь к карте (.jpg/.png/.tif)
  --legend  путь к легенде (опционально; если не указан — ищем внутри карты)
  --sheets  советские номенклатурные листы через запятую: M-43-В,M-42-Г,...
  --output  папка для результатов

Выход: Shapefile + GeoJSON в папке --output.
"""

import argparse
import logging
import os
import sys

from pipeline import georeference, tiling, preprocess, legend, segment, export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def run(args) -> None:
    debug_dir   = os.path.join(args.output, "debug")
    swatches_dir = os.path.join(args.output, "swatches")
    os.makedirs(args.output,   exist_ok=True)
    os.makedirs(debug_dir,     exist_ok=True)
    os.makedirs(swatches_dir,  exist_ok=True)

    if not os.path.exists(args.map):
        log.error(f"Карта не найдена: {args.map}")
        sys.exit(1)

    embedded_legend = args.legend is None or args.legend == args.map
    if not embedded_legend and not os.path.exists(args.legend):
        log.error(f"Легенда не найдена: {args.legend}")
        sys.exit(1)

    # ── Шаг 1: Номенклатурные листы → координаты углов ────────────────────
    log.info("═" * 60)
    log.info("ШАГ 1 — Номенклатурные листы → координаты углов")
    if args.sheets.endswith(".json"):
        import json as _json
        with open(args.sheets, encoding="utf-8") as _f:
            sheet_names = _json.load(_f)["sheets"]
    else:
        sheet_names = [s.strip() for s in args.sheets.split(",") if s.strip()]
    bbox = tiling.sheets_to_bbox(sheet_names)
    log.info(f"    Листы: {sheet_names}")
    log.info(f"    BBox (lat_min, lon_min, lat_max, lon_max): {bbox}")

    # ── Шаг 2: Тайлинг — листы 1:100 000 ─────────────────────────────────
    log.info("═" * 60)
    log.info("ШАГ 2 — Тайлинг: разбивка на листы 1:100 000")
    tiles = tiling.generate_tiles(bbox)
    log.info(f"    Листов 1:100 000: {len(tiles)}")

    # ── Шаг 3: Улучшение качества изображения ─────────────────────────────
    log.info("═" * 60)
    log.info("ШАГ 3 — Предобработка: шумоподавление и контраст")
    enhanced_map = preprocess.enhance(args.map, debug_dir)
    legend_src = args.legend if not embedded_legend else None
    if legend_src:
        enhanced_legend = preprocess.enhance(legend_src, debug_dir, suffix="_legend")
    else:
        enhanced_legend = enhanced_map

    # ── Шаг 4: Детекция границы карты ────────────────────────────────────
    log.info("═" * 60)
    if args.no_crop:
        log.info("ШАГ 4 — Граница карты: автодетекция (--no-crop)")
    else:
        log.info("ШАГ 4 — Граница карты: интерактивный кроп")
    map_image, transform, map_rect, polygon = georeference.build_transform(
        enhanced_map, bbox, debug_dir, interactive=not args.no_crop
    )
    log.info(f"    Карта обрезана: {map_image.shape[1]}×{map_image.shape[0]}px")
    _save_tiles_grid(debug_dir, bbox, tiles)

    # ── Шаг 5: Извлечение легенды ─────────────────────────────────────────
    log.info("═" * 60)
    if embedded_legend:
        log.info("ШАГ 5 — Легенда встроена: извлекаем из области вне рамки карты")
    else:
        log.info("ШАГ 5 — Легенда передана отдельным файлом")
    legend_entries = legend.extract_legend(
        enhanced_legend if not embedded_legend else enhanced_map,
        map_path=enhanced_map,
        map_rect=map_rect,
        map_polygon=polygon,
        output_dir=args.output,   # swatches/ пишутся в корень output
        debug_dir=debug_dir,
    )
    log.info(f"    Записей в легенде: {len(legend_entries)}")

    # ── Шаг 6: OCR текста в легенде — название фракций ───────────────────
    # (уже выполнен внутри legend.extract_legend, swatches сохранены)
    log.info("═" * 60)
    log.info("ШАГ 6 — Фракции из легенды (OCR Tesseract rus+eng)")
    for e in legend_entries[:5]:
        log.info(f"    • [{e['code']}] {e['name']}  {e['color_hex']}")
    if len(legend_entries) > 5:
        log.info(f"    ... и ещё {len(legend_entries) - 5}")

    # ── Шаг 7: Сегментация карты по легенде ──────────────────────────────
    log.info("═" * 60)
    log.info("ШАГ 7 — Сегментация карты: сопоставление с легендой")
    segments = segment.segment_map(map_image, legend_entries, debug_dir)
    log.info(f"    Формаций найдено: {len(segments)}")

    # ── Шаг 8: Определение листа для каждой фракции + экспорт ────────────
    log.info("═" * 60)
    log.info("ШАГ 8 — Экспорт: GeoJSON по листам + combined.shp")
    export.export_tiles(segments, transform, tiles, args.output)
    log.info(f"    Результаты → {args.output}/")
    log.info("═" * 60)
    log.info("ГОТОВО")


def _save_tiles_grid(debug_dir: str, bbox: tuple, tiles: list) -> None:
    import cv2, numpy as np, os
    border_path = os.path.join(debug_dir, "border_detection.jpg")
    if not os.path.exists(border_path):
        return
    img = cv2.imread(border_path)
    if img is None:
        return
    h, w = img.shape[:2]
    lat_min, lon_min, lat_max, lon_max = bbox

    def to_px(lat, lon):
        x = int((lon - lon_min) / (lon_max - lon_min) * w)
        y = int((lat_max - lat) / (lat_max - lat_min) * h)
        return x, y

    lats = sorted({t["bbox"][0] for t in tiles} | {t["bbox"][2] for t in tiles})
    lons = sorted({t["bbox"][1] for t in tiles} | {t["bbox"][3] for t in tiles})

    for lat in lats:
        cv2.line(img, to_px(lat, lon_min), to_px(lat, lon_max), (0, 0, 200), 1)
    for lon in lons:
        cv2.line(img, to_px(lat_max, lon), to_px(lat_min, lon), (0, 0, 200), 1)

    out = os.path.join(debug_dir, "tiles_grid.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log.info(f"    Сетка тайлов → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TerraSoviet — векторизация советских геологических карт"
    )
    parser.add_argument("--map",    required=True, help="Путь к карте (.jpg/.png)")
    parser.add_argument("--legend", default=None,
                        help="Путь к легенде. Если не указан — ищем внутри карты.")
    parser.add_argument("--sheets", required=True,
                        metavar="M-43-В,M-42-Г,...",
                        help="Советские номенклатурные листы через запятую")
    parser.add_argument("--output", required=True, help="Папка для GeoJSON/SHP")
    parser.add_argument("--no-crop", action="store_true",
                        help="Автодетекция границы карты по яркостному профилю. "
                             "По умолчанию открывается интерактивное окно для "
                             "разметки кликами (сохраняется data/polygon_points.txt).")
    run(parser.parse_args())
