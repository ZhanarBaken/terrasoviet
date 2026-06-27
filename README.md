# TerraSoviet Data Rescue — Трек 2: Векторизация карт

Автоматизированный пайплайн для векторизации советских структурно-формационных карт.
Вход: растровое изображение карты + легенды. Выход: GeoJSON/SHP с геологическими полигонами.

---

## Быстрый старт

```bash
pip install -r requirements.txt
brew install tesseract tesseract-lang   # macOS, для OCR кодов
# Ubuntu: apt-get install tesseract-ocr tesseract-ocr-rus

python main.py \
  --map    data/map.jpg \
  --legend data/legenda.jpg \
  --output results/ \
  --bbox   46,69,50,75
```

`--bbox` = `lat_min,lon_min,lat_max,lon_max` в WGS84.

---

## Результаты

```
results/
├── combined.geojson        # все полигоны (312 объектов)
├── combined.shp            # то же в Shapefile (ESRI)
├── M-43-97.geojson         # по советским листам 1:100 000
├── M-43-98.geojson
├── ...                     # 144 листа в bbox
└── errors.log              # что не обработалось
```

Каждый объект в GeoJSON:
```json
{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [...] },
  "properties": {
    "formation": "Гранитовая (C₁, балхашский комплекс)",
    "color_hex": "#e8c4a0",
    "sheet":     "M-43-97"
  }
}
```

---

## Как работает пайплайн

### 1. Геопривязка (`pipeline/georeference.py`)
- Автоматически обрезает рамку карты (Otsu + контурный анализ, ≤6 вершин)
- Строит аффинную матрицу пиксель → WGS84 через `rasterio.from_bounds`
- Точность: ±0.0004° по углам

### 2. Тайловая сетка (`pipeline/tiling.py`)
- Генерирует советские листы 1:100 000 (20'×30') в пределах bbox
- Формула номенклатуры: зона 1:1М → лист 12×12 = 144 листа
- Пример: `(49.5°N, 71.0°E)` → `M-42-95`

### 3. Легенда (`pipeline/legend.py`)
- Находит цветные образцы в legenda.jpg (HSV saturation > 80)
- OCR геологических кодов справа от образца (pytesseract)
- Fallback: KMeans(30) по карте — работает без чистой легенды

### 4. Сегментация — Watershed (`pipeline/segment.py`)
- Граница = любые тёмные непрерывные линии (порог gray < 80)
- Цветовые зоны как seeds для Watershed (помощник, не основной критерий)
- Ч/б карты: distance transform вместо цвета
- OCR кода внутри каждого региона → fuzzy-match к `data/legend_codes.json` (edit distance ≤ 2)
- Результат: ~310 регионов (вместо 13 000+ при наивной цветовой сегментации)

### 5. Экспорт (`pipeline/export.py`)
- Пиксельные контуры → WGS84 полигоны через rasterio
- Clip по bbox тайла (shapely)
- GeoJSON + Shapefile

---

## Структура репо

```
terrasoviet/
├── data/
│   ├── map.jpg             # Структурно-Формационная схема 1:500 000
│   ├── legenda.jpg         # Условные обозначения
│   └── legend_codes.json   # ручной словарь: код → название формации
├── main.py                 # CLI
├── requirements.txt
└── pipeline/
    ├── georeference.py
    ├── tiling.py
    ├── legend.py
    ├── segment.py
    └── export.py
```

---

## Советские листы 1:100 000

Карта покрывает 4 квадранта 1:500 000 (суммарно 4°×6°):

| Квадрант | Широта | Долгота |
|---|---|---|
| M-42-Г | 48–50°N | 69–72°E |
| M-43-В | 48–50°N | 72–75°E |
| L-42-Б | 46–48°N | 69–72°E |
| L-43-А | 46–48°N | 72–75°E |

В каждом квадранте 36 листов 1:100 000. Итого **144 листа**.

---

## Зависимости

```
opencv-python>=4.8.0
numpy>=1.24.0
shapely>=2.0.0
geopandas>=0.14.0
pyproj>=3.6.0
pytesseract>=0.3.10
Pillow>=10.0.0
scikit-learn>=1.3.0
rasterio>=1.3.0
```

---

## Масштабируемость

Пайплайн не привязан к конкретным файлам — все пути через аргументы CLI:

```bash
python main.py --map другая_карта.jpg --legend другая_легенда.jpg \
               --output out/ --bbox 44,65,48,71
```

Работает с любой советской геологической картой, где известны угловые координаты.
