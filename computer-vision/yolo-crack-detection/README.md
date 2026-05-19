# YOLO Crack Detection

Детектирование трещин в видеопотоке с использованием YOLOv8 Classification.

##  Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Полный pipeline одним запуском
python main.py --epochs 1 --batch-size 4 --device cpu

# 3. Старый CLI для обучения тоже остаётся доступным
python train.py --all --epochs 1 --batch-size 4
```

##  Основные команды

### Подготовка датасета для YOLO
```bash
python train.py --prepare-data
```

### Полный запуск через `main.py`
```bash
python main.py --epochs 50 --batch-size 32 --device cuda:0
python main.py --epochs 7 --batch-size 32 --device cpu --run-name my_test
```

Результаты сохраняются в:
```bash
outputs/run_YYYYMMDD_HHMMSS/             (без --run-name)
outputs/my_test_YYYYMMDD_HHMMSS/         (с --run-name="my_test")
```

Внутри будут:
- `best_model.pt`
- `metrics.json`
- `classification_metrics.json`
- `training_report.txt`
- `training_metrics.png`
- `class_distribution.png` (если выполнялась подготовка данных)
- `run_manifest.json`
- папка `ultralytics_run/` с артефактами обучения

### Обучение модели
```bash
python train.py --train --epochs 50
```

##  Структура проекта

```
yolo-crack-detection/
├── main.py                    # Полный pipeline обучения
├── train.py                   # Обучение модели
├── requirements.txt           # Зависимости
├── configs/
│   └── train_config.yaml      # Конфиг
├── src/
│   ├── data_preparation/      # Конвертация данных
│   ├── models/                # YOLO модель
│   ├── inference/             # Обработка видео
│   └── utils/                 # Утилиты
├── data/
│   ├── videos/                # Создаваемые видео
│   └── dataset/               # YOLO датасет
├── outputs/                   # Обученные модели
└── results/                   # Результаты старых запусков / видео
```

##  Workflow

1. Подготовка датасета
2. Обучение YOLOv8n-cls модели
3. Автоматический расчёт метрик
4. Сохранение всего результата в `outputs/run_<timestamp>/`

##  Конфигурация

Отредактируйте `configs/train_config.yaml`:
- `training.epochs` - количество эпох
- `training.batch_size` - размер батча
- `training.device` - CUDA или CPU
- `inference.confidence_threshold` - порог срабатывания

## ✨ Особенности

- ✅ YOLOv8n Classification (быстро и легко)
- ✅ Работает с видеопотоком frame-by-frame
- ✅ Автоматическая отчётность о найденных трещинах
- ✅ Визуализация результатов (красный/зелёный)
- ✅ Использует датасеты DeepCrack и SDNET

