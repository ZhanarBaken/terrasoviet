import json
import os
import numpy as np


def export_geojson(segments: dict, output_path: str) -> None:
    """
    Вход:
        segments    — dict из segment()
        output_path — путь для сохранения .geojson файла
    Выход:
        сохраняет GeoJSON файл на диск

    Формат GeoJSON:
        каждый layer  → Feature с geometry типа Polygon
        каждый fault  → Feature с geometry типа LineString
    """
    features = []

    for layer in segments.get("layers", []):
        contour = layer.get("mask")
        if contour is None:
            continue
        # TODO: конвертировать маску → контур → GeoJSON Polygon
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # coords = contour.reshape(-1, 2).tolist()
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": []  # TODO: заполнить координатами
            },
            "properties": {
                "type": "layer",
                "name": layer.get("name", "unknown")
            }
        }
        features.append(feature)

    for fault in segments.get("faults", []):
        contour = fault.get("contour")
        if contour is None:
            continue
        # TODO: конвертировать контур → GeoJSON LineString
        coords = contour.reshape(-1, 2).tolist() if contour is not None else []
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "type": "fault"
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
