"""
Шаг 7: Сегментация карты — поиск фракций по легенде.

Алгоритм (LAB nearest-centroid):
  1. Конвертируем карту в LAB (перцептивное цветовое пространство)
  2. Для каждого пикселя находим ближайшую легенду-запись по LAB-расстоянию
  3. Каждый пиксель принадлежит ровно одной фракции (нет перекрытий)
  4. Фон (бумага, линии, текст) исключаем по L и насыщенности
  5. findContours → полигоны фракций
"""

import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MIN_AREA_PX = 500


def segment_map(image: np.ndarray, legend: list[dict],
                output_dir: str = None) -> list[dict]:
    log.info(f"    Карта: {image.shape[1]}×{image.shape[0]}px, "
             f"записей в легенде: {len(legend)}")

    # Собираем легенду с LAB-цветами
    centers = []
    valid_legend = []
    for entry in legend:
        bgr = _hex_to_bgr(entry.get("color_hex", ""))
        if bgr is None:
            continue
        lab_color = cv2.cvtColor(
            np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        centers.append(lab_color)
        valid_legend.append(entry)

    if not centers:
        log.warning("    Нет валидных цветов в легенде")
        return []

    centers = np.array(centers, dtype=np.float32)  # (N, 3)
    log.info(f"    Цветов легенды: {len(centers)}")

    # LAB пикселей карты
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3)  # (H*W, 3)

    # Маска фона: бумага (очень светлая), текст/линии (очень тёмная),
    # ненасыщенные серые (изолинии, надписи)
    L = pixels[:, 0]
    A = pixels[:, 1]
    B = pixels[:, 2]
    bg_mask = (
        (L > 230) |                                          # бумага
        (L < 25) |                                           # чёрные линии/текст
        ((np.abs(A - 128) < 7) & (np.abs(B - 128) < 7))    # серые эл-ты
    )

    # Nearest-centroid: 38 проходов по 66М пикселей
    min_sq = np.full(len(pixels), np.inf, dtype=np.float32)
    labels = np.full(len(pixels), -1, dtype=np.int32)

    for i, center in enumerate(centers):
        sq_d = np.sum((pixels - center) ** 2, axis=1)
        better = sq_d < min_sq
        min_sq[better] = sq_d[better]
        labels[better] = i

    labels[bg_mask] = -1
    label_map = labels.reshape(h, w).astype(np.int32)

    # Контуры для каждой фракции
    results = []
    for i, entry in enumerate(valid_legend):
        mask = (label_map == i).astype(np.uint8) * 255
        mask = _clean_mask(mask)
        contours = _find_contours(mask)
        log.info(f"    [{i+1}/{len(valid_legend)}] "
                 f"{(entry.get('code') or entry.get('name',''))[:30]} "
                 f"{entry.get('color_hex','')} — контуров: {len(contours)}")
        if not contours:
            continue
        results.append({
            "name":      entry.get("name", ""),
            "code":      entry.get("code", ""),
            "color_hex": entry.get("color_hex", ""),
            "contours":  contours,
        })

    log.info(f"    Итого формаций: {len(results)}")

    if output_dir:
        _save_preview(image, results, output_dir)
        _save_label_preview(image, label_map, valid_legend, output_dir)

    return results


# ── Утилиты ───────────────────────────────────────────────────────────────

def _hex_to_bgr(hex_str: str):
    """#RRGGBB → (B, G, R). Возвращает None если не парсится."""
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)
    except ValueError:
        return None


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def _find_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for c in contours:
        if cv2.contourArea(c) < _MIN_AREA_PX:
            continue
        simplified = cv2.approxPolyDP(c, epsilon=3.0, closed=True)
        result.append(simplified)
    return result


# ── Превью ────────────────────────────────────────────────────────────────

def _save_preview(image: np.ndarray, results: list[dict], output_dir: str) -> None:
    vis = image.copy()
    for seg in results:
        hex_col = seg.get("color_hex", "")
        bgr = _hex_to_bgr(hex_col) or (0, 200, 0)
        cv2.drawContours(vis, seg["contours"], -1, bgr, 2)
    path = os.path.join(output_dir, "segments_preview.jpg")
    h, w = vis.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    cv2.imwrite(path, cv2.resize(vis, (int(w * scale), int(h * scale))))
    log.info(f"    Превью сегментации → {path}")


def _save_label_preview(image: np.ndarray, label_map: np.ndarray,
                        legend: list[dict], output_dir: str) -> None:
    """Закрашиваем каждый пиксель цветом его фракции."""
    vis = np.ones_like(image) * 255
    for i, entry in enumerate(legend):
        bgr = _hex_to_bgr(entry.get("color_hex", ""))
        if bgr is None:
            continue
        vis[label_map == i] = bgr
    path = os.path.join(output_dir, "label_map.jpg")
    h, w = vis.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    cv2.imwrite(path, cv2.resize(vis, (int(w * scale), int(h * scale))))
    log.info(f"    Label map → {path}")
