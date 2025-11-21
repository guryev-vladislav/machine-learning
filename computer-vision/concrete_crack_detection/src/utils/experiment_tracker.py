# src/utils/experiment_tracker.py
import json
import logging
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models_base = Path("models")
        self.results_base = Path("results")

    def get_next_model_version(self):
        try:
            if not self.models_base.exists():
                return "model_v1"

            existing = [d.name for d in self.models_base.iterdir()
                        if d.is_dir() and d.name.startswith("model_v")]
            if not existing:
                return "model_v1"

            versions = [int(v.split('_v')[1]) for v in existing if v.split('_v')[1].isdigit()]
            next_version = max(versions) + 1 if versions else 1
            return f"model_v{next_version}"

        except Exception as e:
            self.logger.error(f"Failed to determine model version: {e}")
            return "model_v1"

    def create_experiment_folders(self, model_version):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_folder = f"result_{model_version}_{timestamp}"

            model_path = self.models_base / model_version
            result_path = self.results_base / result_folder

            model_path.mkdir(parents=True, exist_ok=True)
            result_path.mkdir(parents=True)
            (result_path / "training_data").mkdir()
            (result_path / "plots").mkdir()
            (result_path / "test_results").mkdir()
            (result_path / "plots" / "prediction_samples").mkdir()

            self.logger.info(f"Created experiment folders: {model_version}")
            return str(model_path), str(result_path)

        except Exception as e:
            self.logger.error(f"Failed to create experiment folders: {e}")
            raise

    def save_metadata(self, result_path, config):
        try:
            metadata = {
                "experiment_id": Path(result_path).name,
                "timestamp": datetime.now().isoformat(),
                "config": config
            }

            meta_file = Path(result_path) / "metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved metadata to {meta_file}")

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise