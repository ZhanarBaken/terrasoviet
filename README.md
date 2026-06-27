# TerraSoviet Data Rescue

Автоматизированный pipeline для векторизации советских геологических карт.
Трек 2 — Хакатон TerraSoviet Data Rescue, 24 часа.

---

## Что делает pipeline

1. Берёт растровую карту (`map.jpg`) и легенду (`legenda.jpg`)
2. Геопривязывает карту по угловым координатам
3. Сегментирует геологические регионы — по **цвету** + **текстовым кодам** внутри (OCR)
4. Классифицирует каждый регион по легенде (формация, код)
5. Делит результат по советским листам 1:100 000 (20'×30')
6. Экспортирует GeoJSON — один файл на лист + `combined.geojson`

---

## Запуск

```bash
pip install -r requirements.txt

python main.py \
  --map  data/map.jpg \
  --legend data/legenda.jpg \
  --output results/ \
  --bbox 46,69,50,75
```

`--bbox` = `lat_min,lon_min,lat_max,lon_max` в WGS84.
Для нашей карты: `46,69,50,75` (46–50°N, 69–75°E).

Результаты появятся в `results/`:
```
results/
├── combined.geojson      # все объекты вместе
├── M-43-97.geojson       # по листам 1:100 000
├── M-43-98.geojson
├── ...
└── errors.log            # что не обработалось (если есть)
```

---

## Структура репо

```
terrasoviet/
├── data/
│   ├── map.jpg            # Структурно-Формационная схема 1:500 000
│   └── legenda.jpg        # Условные обозначения к ней
├── main.py                # точка входа (CLI)
├── requirements.txt
└── pipeline/
    ├── georeference.py    # пиксель ↔ WGS84 (ЗЕЛЁНЫЙ)
    ├── tiling.py          # советская номенклатура листов (ЗЕЛЁНЫЙ)
    ├── legend.py          # извлечение палитры из легенды (ЗЕЛЁНЫЙ)
    ├── segment.py         # сегментация карты (ФИОЛЕТОВЫЙ)
    └── export.py          # экспорт GeoJSON (ОРАНЖЕВЫЙ)
```

---

## Задачи команды

### ЗЕЛЁНЫЙ — `georeference.py`, `tiling.py`, `legend.py` ✅

Уже реализовано:
- `georeference.build_transform(map_path, bbox)` — обрезает рамку карты, строит аффинную матрицу пиксель→WGS84
- `tiling.generate_tiles(bbox)` — генерирует 144 листа 1:100 000 по советской номенклатуре
- `legend.extract_legend(legend_path)` — находит цветные образцы в легенде, OCR названий

Проверено: углы карты попадают точно в `69°E / 50°N` и `75°E / 46°N`.

---

### ФИОЛЕТОВЫЙ — `pipeline/segment.py`

**Интерфейс уже задан, нужно улучшить логику внутри.**

Функция:
```python
def segment_map(image: np.ndarray, legend: list[dict]) -> list[dict]:
```

Вход:
- `image` — BGR карта (numpy array) после `georeference.build_transform()`
- `legend` — список записей из `legend.extract_legend()`, каждая запись:
  ```python
  {
    "name":      "Гранитовая (C₁, балхашский комплекс)",
    "code":      "C₁",
    "color_hex": "#ff9a7b",
    "hsv_lower": [0, 80, 100],   # нижняя граница HSV
    "hsv_upper": [15, 200, 255], # верхняя граница HSV
  }
  ```

Выход:
```python
[{
    "name":      "Гранитовая (C₁, ...)",
    "color_hex": "#ff9a7b",
    "contours":  [np.ndarray, ...],  # пиксельные контуры (cv2.findContours формат)
}]
```

Текущая реализация (заглушка) — цветовая сегментация через `cv2.inRange` + OCR кода внутри региона.
**Твоя задача**: подобрать параметры, добавить SAM для уточнения границ, проверить на карте.

Советы:
- HSV диапазоны легенды могут не точно совпадать с картой — добавь tolerance ±15 по H, ±80 по S/V
- Маленькие контуры (< 200 px²) выбрасывать
- `cv2.approxPolyDP(c, epsilon=3, closed=True)` — упрощение контуров

---

### ОРАНЖЕВЫЙ — `pipeline/export.py`

**Интерфейс уже задан, нужно улучшить + добавить Shapefile.**

Функция:
```python
def export_tiles(segments, transform, tiles, output_dir):
```

Вход:
- `segments` — из `segment.segment_map()`
- `transform` — Affine объект из `georeference.build_transform()`
- `tiles` — список `[{"name": "M-43-97", "bbox": (lat_min, lon_min, lat_max, lon_max)}, ...]`
- `output_dir` — куда писать

Текущий выход: GeoJSON на каждый лист + `combined.geojson`.
**Твоя задача**:
1. Добавить экспорт Shapefile (через `geopandas.to_file(..., driver='ESRI Shapefile')`)
2. Добавить overlay визуализацию (карта + маски) в `results/preview/`
3. Убедиться что `errors.log` пишется при любой ошибке

Properties в каждом GeoJSON Feature:
```json
{
  "formation": "Гранитовая (C₁, балхашский комплекс)",
  "color_hex": "#ff9a7b",
  "sheet":     "M-43-97"
}
```

---

## Советские листы 1:100 000

Карта охватывает 4 квадранта 1:500 000:
| Квадрант | Широта | Долгота |
|---|---|---|
| M-42-Г | 48–50°N | 69–72°E |
| M-43-В | 48–50°N | 72–75°E |
| L-42-Б | 46–48°N | 69–72°E |
| L-43-А | 46–48°N | 72–75°E |

В каждом квадранте — 36 листов 1:100 000. Итого: **144 листа**.
Размер листа: 20' широты × 30' долготы.

---

## Зависимости

```
opencv-python>=4.8.0
numpy>=1.24.0
shapely>=2.0.0
geopandas>=0.14.0
pyproj>=3.6.0
pytesseract>=0.3.10    # нужен Tesseract + rus языковой пакет
Pillow>=10.0.0
scikit-learn>=1.3.0
rasterio>=1.3.0
```

Установка Tesseract (macOS):
```bash
brew install tesseract tesseract-lang
```

Установка Tesseract (Ubuntu):
```bash
apt-get install tesseract-ocr tesseract-ocr-rus
```

---

## Дедлайн

Заморозка кода в **12:00**. После — никаких коммитов.
Судьи клонируют репо и запускают `python main.py --map ... --legend ... --output ... --bbox ...` на своём датасете.
**Никаких хардкод путей в коде.**
