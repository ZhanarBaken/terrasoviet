import argparse
import logging
import os
import sys

from pipeline import georeference, tiling, legend, segment, export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_bbox(s: str) -> tuple:
    """'46,69,50,75' → (46.0, 69.0, 50.0, 75.0)"""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "bbox: lat_min,lon_min,lat_max,lon_max"
        )
    return tuple(parts)


def run(args) -> None:
    if not os.path.exists(args.map):
        log.error(f"Карта не найдена: {args.map}")
        sys.exit(1)
    if not os.path.exists(args.legend):
        log.error(f"Легенда не найдена: {args.legend}")
        sys.exit(1)

    log.info("1/5 Геопривязка карты...")
    map_image, transform = georeference.build_transform(args.map, args.bbox)

    log.info("2/5 Генерация тайловой сетки...")
    tiles = tiling.generate_tiles(args.bbox)
    log.info(f"    Листов 1:100 000: {len(tiles)}")

    log.info("3/5 Извлечение легенды...")
    legend_entries = legend.extract_legend(args.legend, map_path=args.map)
    log.info(f"    Записей в легенде: {len(legend_entries)}")

    log.info("4/5 Сегментация карты (цвет + OCR)...")
    segments = segment.segment_map(map_image, legend_entries)
    log.info(f"    Формаций: {len(segments)}")

    log.info("5/5 Экспорт GeoJSON...")
    export.export_tiles(segments, transform, tiles, args.output)
    log.info(f"Готово → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TerraSoviet — векторизация советских геологических карт"
    )
    parser.add_argument("--map",    required=True, help="Путь к карте (.jpg/.png/.tif)")
    parser.add_argument("--legend", required=True, help="Путь к легенде (.jpg/.png)")
    parser.add_argument("--output", required=True, help="Папка для GeoJSON")
    parser.add_argument(
        "--bbox", required=True, type=parse_bbox,
        metavar="LAT_MIN,LON_MIN,LAT_MAX,LON_MAX",
        help="Охват в WGS84. Пример: 46,69,50,75",
    )
    run(parser.parse_args())
