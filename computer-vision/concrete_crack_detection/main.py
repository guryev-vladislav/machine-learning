import logging
from src.utils.config import Config
from src.training.pipeline import TrainingPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


def main():
    # 1. Загрузка конфигурации
    config = Config()

    # 2. Инициализация пайплайна
    pipeline = TrainingPipeline(config)

    # 3. Путь к тестовому изображению (если есть)
    # Замените на реальный путь к картинке для проверки
    sample_img = config.DEEPCRACK_PATH / "rgb/001.jpg"

    # 4. Запуск полного цикла (Обучение + Метрики + Тест)
    # ВАЖНО: имя метода теперь run_full_experiment
    pipeline.run_full_experiment(test_image=sample_img)


if __name__ == "__main__":
    main()