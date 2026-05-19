# YOLO Crack Detection - Model Documentation

## Обзор проекта

Этот проект использует **YOLOv8n-cls** (nano-классификация) для классификации изображений бетонных дорог на две категории:
- **crack** (трещина)
- **no_crack** (без трещин)

## Архитектура и выбор модели

### Почему YOLOv8n-cls?

1. **Скорость**: Nano версия обеспечивает ~80 FPS на GPU
2. **Точность**: Достаточно высокая точность для практического применения
3. **Размер**: Компактная модель (5-6 МБ) для развертывания на мобильных устройствах
4. **Простота**: Классификация быстрее, чем детектирование для данной задачи

### Альтернативные модели (в проекте)

- `src/models/pretrained/yolo26n-cls.pt` - Старая версия (оставлена для совместимости)
- `src/models/pretrained/yolo26n.pt` - Детекторная версия (для определения bbox)

## Ключевые метрики оценки

### 1. Accuracy (Точность)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Интерпретация**: Доля правильных предсказаний из всех
- **Когда используется**: Хороший начальный показатель, но может быть неточным при дисбалансе классов
- **Целевое значение**: > 95%

### 2. Precision (Точность определения)
```
Precision = TP / (TP + FP)
```
- **Интерпретация**: Из всех предсказанных трещин, сколько реально являются трещинами
- **Когда используется**: Когда FP дорого (ложные срабатывания приводят к ненужному ремонту)
- **Целевое значение**: > 90%

### 3. Recall (Полнота)
```
Recall = TP / (TP + FN)
```
- **Интерпретация**: Из всех реальных трещин, сколько модель нашла
- **Когда используется**: Когда FN дорого (пропущенные трещины приводят к деградации дороги)
- **Целевое значение**: > 90%

### 4. F1-Score (F-мера)
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- **Интерпретация**: Гармоническое среднее точности и полноты
- **Когда используется**: Баланс между Precision и Recall
- **Целевое значение**: > 92%

### 5. ROC AUC (Area Under Curve)
```
AUC = Площадь под кривой ROC
```
- **Интерпретация**: Вероятность того, что модель правильно ранжирует положительный образец выше отрицательного
- **Когда используется**: Оценка порога классификации
- **Целевое значение**: > 97%

### 6. Confusion Matrix
```
┌─────────┬──────────┬──────────┐
│ Metric  │ Predicted│ Predicted│
│         │  Crack   │ No Crack │
├─────────┼──────────┼──────────┤
│ Actual  │    TP    │    FN    │
│ Crack   │          │          │
├─────────┼──────────┼──────────┤
│ Actual  │    FP    │    TN    │
│ No Crack│          │          │
└─────────┴──────────┴──────────┘
```

## Датасет

### Источники данных:
- **DeepCrack** (458 изображений трещин)
- **SDNET2018** (более 2000 изображений)
  - Categories: Decks, Pavements, Walls
  - Subclasses: Cracked, Non-cracked

### Разделение данных:
```
Training Set: 80%
Validation Set: 20%
```

### Предварительная обработка:
- Обычная нормализация ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Аугментация: случайный кроп, зеркальное отражение
- Размер входа: 640x640 (для YOLOv8)

## Визуализация и аналитика

### Созданные файлы визуализации:

1. **Training History** (`plots/training_history.png`)
   - Loss curve (обучающий и валидационный)
   - Accuracy curve (Top-1 / Top-5)

2. **Confusion Matrix** (`plots/confusion_matrix.png`)
   - TP, TN, FP, FN
   - Процентное соотношение

3. **ROC Curve** (`plots/roc_curve.png`)
   - True Positive Rate vs False Positive Rate
   - AUC значение

4. **Precision-Recall Curve** (`plots/pr_curve.png`)
   - Trade-off между Precision и Recall

5. **Model Comparison** (`plots/model_comparison.png`)
   - Сравнение нескольких моделей
   - Bar charts по разным метрикам

6. **Class Distribution** (`plots/class_distribution.png`)
   - Распределение трещин vs без трещин
   - Абсолютные и относительные значения

7. **Comprehensive Report** (`plots/comprehensive_report.png`)
   - Полный отчет о производительности
   - Radar chart метрик
   - Summary таблица

## Использование метрик и визуализации

### Для обучения:

```python
from src.utils.metrics import classification_metrics
from src.utils.visualizer import (
    plot_training_history,
    plot_confusion_matrix,
    plot_comprehensive_report
)

# Вычисляем метрики
metrics = classification_metrics(y_true, y_scores)

# Визуализируем результаты
plot_training_history(history, 'plots/training_history.png')
plot_confusion_matrix(metrics['confusion_matrix'], 'plots/confusion_matrix.png')
plot_comprehensive_report(metrics, 'plots/report.png')
```

### Для инферентота:

```python
from src.models.yolo_crack_detector import YOLOCrackDetector
from src.utils.visualizer import plot_detection_results

detector = YOLOCrackDetector()
results = detector.predict('image.jpg')

# Визуализируем детектирования
detections = [...]
plot_detection_results('image.jpg', detections, 'result.png')
```

## Результаты на тестовом наборе

### Ожидаемые значения:

| Метрика | Значение |
|---------|----------|
| Accuracy | > 95% |
| Precision | > 90% |
| Recall | > 90% |
| F1-Score | > 92% |
| ROC AUC | > 97% |

### Временные показатели:

| Операция | Время |
|----------|-------|
| Инфернс (1 image) | ~12-15 ms |
| Инфернс (1 fps video @ 10 fps) | ~100 ms per frame |
| Training (50 epochs) | ~2-3 часа на GPU |

## Структура выходных файлов

```
outputs/
├── plots/
│   ├── yolo_curves.png               # История обучения YOLO
│   ├── training_history.png          # История обучения
│   ├── confusion_matrix.png           # Матрица ошибок
│   ├── roc_curve.png                  # ROC кривая
│   ├── pr_curve.png                   # Precision-Recall
│   ├── model_comparison.png           # Сравнение моделей
│   ├── class_distribution.png         # Распределение классов
│   └── comprehensive_report.png       # Полный отчет
├── best_model.pt                      # Лучшая модель
├── training_metrics.json              # Метрики в JSON
└── training_metrics.txt               # Метрики в текст
```

## Для дипломной работы

### Рекомендуемые включения:

1. **Теоретическая часть**:
   - Описание YOLOv8 архитектуры
   - Объяснение полярных метрик
   - Сравнение Precision vs Recall trade-off

2. **Экспериментальная часть**:
   - Results table со всеми метриками
   - Графики (training history, confusion matrix, ROC curve)
   - Анализ ошибок (false positives, false negatives)

3. **Практическое применение**:
   - Time complexity: O(1) на изображение
   - Memory: ~100MB модель + inference buffer
   - Deployment: можно запустить на мобильных устройствах

## Конфигурация обучения

```yaml
training:
  device: "cuda:0"
  epochs: 50
  batch_size: 32
  imgsz: 640
  learning_rate: 0.001
  patience: 10
  workers: 4
  task: "classify"
  model: "yolov8n-cls"
```

## Дополнительные ресурсы

- [YOLOv8 Документация](https://docs.ultralytics.com)
- [DeepCrack Dataset](https://github.com/yhlleo/DeepCrack)
- [SDNET2018 Dataset](https://digitalrepository.unm.edu/cee_datasets)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---
**Дата создания**: 2026-05-04
**Версия**: 1.0
**Автор**: Machine Learning Research Group
