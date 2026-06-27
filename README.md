# TerraSoviet Data Rescue

Автоматизированный pipeline для векторизации советских геологических карт.

## Задача

Трек 2: извлечение геологических объектов (разломы, слои) из растровых исторических карт и экспорт в GeoJSON/Shapefile.

## Запуск

```bash
pip install -r requirements.txt
python main.py --input ./data --output ./results
```

## Структура проекта

```
terrasoviet/
├── main.py               # точка входа, запускает pipeline
├── requirements.txt
├── README.md
└── modules/
    ├── preprocess.py     # обрезка рамки, denoising, классификация типа
    ├── segment.py        # HSV сегментация, детекция линий разломов
    └── export.py         # экспорт в GeoJSON
```

## Pipeline

```
Входное изображение
    → preprocess()       # обрезка + denoising
    → classify_map()     # "color" | "bw" | "composite"
    → segment()          # маски слоёв + контуры разломов
    → export_geojson()   # сохранение .geojson
```

## Команда

TerraSoviet Hackathon 2024 — Агадырская ГПП
