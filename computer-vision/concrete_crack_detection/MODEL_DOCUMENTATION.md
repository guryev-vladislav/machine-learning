# Concrete Crack Detection - Model Documentation

## Обзор проекта

Проект использует **двухэтапную архитектуру**:
1. **UNet** для сегментации трещин (локализация)
2. **Simple CNN** для классификации (бинарная классификация трещина/без трещины)

## Архитектура моделей

### UNet (U-shaped Network) для сегментации

**Назначение**: Точное локализование пикселей с трещинами

**Архитектура**:
```
Input (256x256x3)
    ↓
Encoder: Conv → ReLU → MaxPool (4 уровня)
    ↓
Bottleneck: Conv → ReLU
    ↓
Decoder: UpSample → Concatenate → Conv (4 уровня)
    ↓
Output (256x256x1) - Binary Segmentation Mask
```

**Входы**: RGB изображения 256x256
**Выходы**: Бинарная маска 256x256 (0 = фон, 1 = трещина)

**Функция потерь**: Combined Loss
- Dice Loss (за точность границ)
- BCE Loss (за пиксельную классификацию)
- Position Weight: 8.0 (важнейший параметр для дисбаланса)

### Simple CNN для классификации

**Назначение**: Финальная классификация изображения

**Архитектура**:
```
Input (256x256x3 или маска от UNet)
    ↓
Conv3×3 (32 filters) → ReLU → MaxPool
    ↓
Conv3×3 (64 filters) → ReLU → MaxPool
    ↓
Conv3×3 (128 filters) → ReLU → MaxPool
    ↓
Flatten
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(1) → Sigmoid
    ↓
Binary Output: [no_crack, crack]
```

**Входы**: RGB изображения 256x256 или маски от UNet
**Выходы**: Вероятность наличия трещины [0, 1]

**Функция потерь**: BCEWithLogitsLoss
- Position Weight: 4.0 (для дисбаланса классов)

## Ключевые метрики для сегментации (UNet)

### 1. IoU (Intersection over Union) / Jaccard Index
```
IoU = |A ∩ B| / |A ∪ B|
```
- **Интерпретация**: Отношение пересечения и объединения предсказанной и истинной маски
- **Диапазон**: [0, 1] (1 = идеально)
- **Когда используется**: Основная метрика для задач сегментации
- **Целевое значение**: > 0.75

### 2. Dice Coefficient (F1 для сегментации)
```
Dice = 2|A ∩ B| / (|A| + |B|)
```
- **Интерпретация**: Схожесть двух множеств (особенно хороша при дисбалансе)
- **Диапазон**: [0, 1] (1 = идеально)
- **Когда используется**: Когда классы сильно дисбалансированы
- **Целевое значение**: > 0.80

### 3. Pixel Accuracy
```
PixelAcc = (TP + TN) / (TP + TN + FP + FN)
```
- **Интерпретация**: Доля правильно классифицированных пикселей
- **Диапазон**: [0, 1] (1 = идеально)
- **Когда используется**: Общая оценка качества
- **Целевое значение**: > 95%

## Ключевые метрики для классификации (CNN)

### 1. Accuracy
```
Accuracy = (TP + TN) / Total
```
- **Целевое значение**: > 90%

### 2. Precision
```
Precision = TP / (TP + FP)
```
- **Интерпретация**: Из предсказанных трещин, сколько реальных
- **Целевое значение**: > 85%

### 3. Recall
```
Recall = TP / (TP + FN)
```
- **Интерпретация**: Из всех реальных трещин, сколько найдено
- **Целевое значение**: > 85%

### 4. F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- **Целевое значение**: > 87%

## Датасет

### Источники:
- **DeepCrack**: 458 пар (RGB + маска)
- **SDNET2018**: 2+ тысячи изображений
  - Categories: Decks, Pavements, Walls
  - Subclasses: Cracked, Non-cracked

### Разделение:
```
Training: 80%
Validation: 20%
```

### Предварительная обработка:
- ImageNet нормализация: µ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]
- Аугментация (для обучения):
  - RandomResizedCrop(0.5-1.0 scale)
  - HorizontalFlip (p=0.5)
  - VerticalFlip (p=0.5)
- Размер входа: 256x256

## Визуализация и аналитика

### Созданные файлы (в `outputs/run_*/plots/`):

1. **training_history.png**
   - Loss curves (train/val)
   - Метрики (train/val)

2. **unet_curves.png**
   - UNet training history
   - Loss и IoU/Dice metric

3. **cnn_curves.png**
   - CNN training history
   - Loss и accuracy metric

4. **result_*.png** (для каждого test image)
   - Multi-scale сравнение:
     - Оригинальное изображение
     - UNet маска (тепловая карта)
     - CNN prediction (confidence)

### Пример визуализации результатов:

```
┌─────────────────────────────────────────┐
│ Input Image (256px)                     │ UNet Mask      │ CNN Verdict
├─────────────────────────────────────────┤
│ Оригинальное RGB                        │ Heatmap        │ crack: 0.92
│                                          │ (красное=тр.)  │
├─────────────────────────────────────────┤
│ Input Image (128px)                     │ UNet Mask      │ CNN Verdict
└─────────────────────────────────────────┘
```

## Использование метрик

### Для обучения:

```python
from src.utils.metrics import calculate_metrics
from src.utils.visualizer import plot_training_history, save_multiscale_comparison

# Во время обучения
outputs = unet_model(images)
target = masks
iou, dice = calculate_metrics(outputs, target)

# После обучения
history = {'train_loss': [...], 'val_loss': [...], ...}
plot_training_history(history, 'plots/unet_curves.png', 'UNet')
```

### Для инфернса:

```python
from src.models.unet import UNet
from src.models.classifier import SimpleCNN
from src.utils.visualizer import save_multiscale_comparison

unet = UNet(n_channels=3, n_classes=1)
cnn = SimpleCNN(n_channels=3, n_classes=1)

# Для image:
unet_mask = unet(image)
is_crack = cnn(unet_mask > 0.5)

# Визуализируем результаты
results = [
    ("256px", unet_mask.numpy(), is_crack.item(), original_image),
    ("128px", ..., ..., ...),
    ("64px", ..., ..., ...),
]
save_multiscale_comparison(results, 'result.png')
```

## Конфигурация обучения

```yaml
training:
  device: "cuda:0"
  batch_size: 16
  num_workers: 0
  learning_rate: 0.0002
  epochs: 50
  early_stopping: 8
  
models:
  unet:
    patch_size: 256
    weights_name: "unet.pth"
    pos_weight: 8.0
    
  cnn:
    image_size: 256
    weights_name: "cnn.pth"
    pos_weight: 4.0
```

## Результаты обучения

### Ожидаемые значения для UNet:

| Метрика | Значение |
|---------|----------|
| IoU | > 0.75 |
| Dice | > 0.80 |
| Pixel Accuracy | > 95% |
| Val Loss | < 0.15 |

### Ожидаемые значения для CNN:

| Метрика | Значение |
|---------|----------|
| Accuracy | > 90% |
| Precision | > 85% |
| Recall | > 85% |
| F1-Score | > 87% |

## Структура выходных файлов

```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── models/
    │   ├── unet.pth              # Веса UNet
    │   └── cnn.pth               # Веса CNN
    ├── plots/
    │   ├── unet_curves.png       # История обучения UNet
    │   ├── cnn_curves.png        # История обучения CNN
    │   └── result_*.png          # Результаты на test images
    └── config_used.yaml          # Конфиг для reproducibility
```

## Оптимизация параметров

### Критические параметры:

1. **POS_WEIGHT_UNET** (8.0)
   - Вес положительного класса для UNet
   - Помогает справиться с дисбалансом (93% нет трещин, 7% трещины)
   - Влияет: выше → больше внимания к трещинам

2. **POS_WEIGHT_CNN** (4.0)
   - Вес для CNN
   - Меньше чем для UNet (CNN видит уже обработанные маски)

3. **EARLY_STOPPING** (8)
   - Число эпох без улучшения перед остановкой
   - Экономит время, предотвращает переобучение

4. **BATCH_SIZE** (16)
   - Компромисс между памятью и градиентами
   - 32 MB на GPU (RTX 3080 имеет 10 GB)

## Для дипломной работы

### Рекомендуемые включения:

1. **Раздел "Методология"**:
   - Описание двухэтапного подхода
   - UNet vs FCN vs DeepLab сравнение
   - Объяснение position weights

2. **Раздел "Результаты"**:
   - Таблица с метриками UNet и CNN
   - Графики training curves
   - Примеры successful и failed cases
   - Анализ: когда модель ошибается?

3. **Раздел "Сравнение методов"**:
   - vs YOLO (детекция)
   - vs простая CNN (без разделения на сегментацию)
   - vs традиционные методы (морфология, фильтры)

4. **Практическое применение**:
   - Performance: ~2 сек на изображение (CPU), ~50ms (GPU)
   - Memory: 50 MB модели
   - Scalability: можно обрабатывать видео в реальном времени

## Отладка и анализ ошибок

### Когда модель ошибается:

1. **False Positives** (предсказала трещину, но ее нет):
   - Обычно: тени, загрязнения, швы
   - Решение: аугментация, threshold tuning

2. **False Negatives** (не обнаружила реальную трещину):
   - Обычно: тонкие трещины, частичная видимость
   - Решение: увеличить pos_weight, больше данных

## Дополнительные ресурсы

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Dice Loss](https://arxiv.org/abs/1606.06650)
- [DeepCrack](https://github.com/yhlleo/DeepCrack)
- [SDNET2018](https://digitalrepository.unm.edu/cee_datasets)

---
**Дата создания**: 2026-05-04
**Версия**: 1.0
**Framework**: PyTorch
**Python**: 3.10+
