import argparse
import logging
import os

from modules import preprocess, classify_map, segment, export_geojson

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def run_pipeline(input_dir: str, output_dir: str) -> None:
    """
    Вход:  папка с картами (input_dir)
    Выход: GeoJSON файлы в output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        log.warning(f"Изображения не найдены в: {input_dir}")
        return

    log.info(f"Найдено {len(image_paths)} карт")

    success, failed = 0, 0

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{stem}.geojson")

        try:
            log.info(f"Обработка: {filename}")

            image = preprocess(image_path)
            map_type = classify_map(image)
            log.info(f"  Тип карты: {map_type}")

            segments = segment(image, map_type)
            export_geojson(segments, output_path)

            log.info(f"  Сохранено: {output_path}")
            success += 1

        except Exception as e:
            log.error(f"  Ошибка при обработке {filename}: {e}")
            failed += 1

    log.info(f"Готово: {success} успешно, {failed} ошибок")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TerraSoviet — векторизация советских геологических карт"
    )
    parser.add_argument(
        "--input", required=True,
        help="Папка с входными изображениями карт"
    )
    parser.add_argument(
        "--output", required=True,
        help="Папка для сохранения GeoJSON файлов"
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
