"""
Экспорт сегментов в GeoJSON — модуль Вайбкодера (оранжевый).

Выход:
  results/{tile_name}.geojson  — один файл на лист 1:100 000
  results/combined.geojson     — всё вместе
  results/errors.log           — что не удалось обработать
"""

import json
import logging
import os

import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.validation import make_valid

log = logging.getLogger(__name__)


def export_tiles(
    segments: list[dict],
    transform,
    tiles: list[dict],
    output_dir: str,
) -> None:
    """
    Вход:
        segments   — из segment.segment_map()
        transform  — Affine из georeference.build_transform()
        tiles      — из tiling.generate_tiles()
        output_dir — куда писать
    """
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    errors = []

    for seg in segments:
        for contour in seg["contours"]:
            try:
                coords = _contour_to_lonlat(contour, transform)
                if len(coords) < 3:
                    continue
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue

                tile_name = _find_tile(poly.centroid.x, poly.centroid.y, tiles)
                feature = {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "formation": seg["name"],
                        "color_hex": seg["color_hex"],
                        "sheet": tile_name,
                    },
                }
                all_features.append(feature)
            except Exception as e:
                errors.append(str(e))
                log.debug(f"Пропущен контур: {e}")

    # Один GeoJSON на тайл
    tile_map: dict[str, list] = {}
    for f in all_features:
        key = f["properties"]["sheet"]
        tile_map.setdefault(key, []).append(f)

    for tile_name, features in tile_map.items():
        _save_geojson(features, os.path.join(output_dir, f"{tile_name}.geojson"))

    # Общий файл
    _save_geojson(all_features, os.path.join(output_dir, "combined.geojson"))

    # Shapefile (опционально, если geopandas доступен)
    _save_shapefile(all_features, os.path.join(output_dir, "combined.shp"))

    # Лог ошибок
    if errors:
        with open(os.path.join(output_dir, "errors.log"), "w") as f:
            f.write("\n".join(errors))

    log.info(
        f"Экспорт завершён: {len(all_features)} объектов, "
        f"{len(tile_map)} тайлов, {len(errors)} ошибок"
    )


def _contour_to_lonlat(contour: np.ndarray, transform) -> list[tuple]:
    from rasterio.transform import xy as rio_xy
    pts = contour.reshape(-1, 2)
    coords = []
    for col, row in pts:
        lon, lat = rio_xy(transform, int(row), int(col))
        coords.append((float(lon), float(lat)))
    return coords


def _find_tile(lon: float, lat: float, tiles: list[dict]) -> str:
    for tile in tiles:
        lat_min, lon_min, lat_max, lon_max = tile["bbox"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return tile["name"]
    return "unknown"


def _save_shapefile(features: list, path: str) -> None:
    try:
        import geopandas as gpd
        from shapely.geometry import shape

        if not features:
            return
        gdf = gpd.GeoDataFrame(
            [f["properties"] for f in features],
            geometry=[shape(f["geometry"]) for f in features],
            crs="EPSG:4326",
        )
        gdf.to_file(path, driver="ESRI Shapefile", encoding="utf-8")
        log.info(f"Shapefile: {path}")
    except Exception as e:
        log.warning(f"Shapefile не сохранён: {e}")


def _save_geojson(features: list, path: str) -> None:
    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    log.debug(f"Сохранено: {path} ({len(features)} объектов)")
