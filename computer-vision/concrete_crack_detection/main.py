import logging
from src.utils.config import Config
from src.training.pipeline import TrainingPipeline


def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Инициализируем пайплайн (теперь без аргументов в скобках)
    pipeline = TrainingPipeline()

    # Путь к тестовой картинке для финального отчета
    # Убедитесь, что этот файл существует в вашей папке данных
    sample_img = "data/external/deepcrack/rgb/001.jpg"

    # Запуск полного цикла обучения и тестов
    pipeline.run_full_experiment(test_image=sample_img)


if __name__ == "__main__":
    main()