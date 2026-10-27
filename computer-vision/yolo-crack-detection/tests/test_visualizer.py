"""Tests for src/utils/visualizer.py."""

import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use('Agg')


class SafeFloatIntTests(unittest.TestCase):
    """Tests for _safe_float and _safe_int helper functions."""

    def test_safe_float_valid(self):
        from src.utils.visualizer import _safe_float

        self.assertEqual(_safe_float('3.14'), 3.14)
        self.assertEqual(_safe_float('42'), 42.0)
        self.assertEqual(_safe_float('0'), 0.0)

    def test_safe_float_invalid(self):
        from src.utils.visualizer import _safe_float

        self.assertIsNone(_safe_float(None))
        self.assertIsNone(_safe_float(''))
        self.assertIsNone(_safe_float('abc'))

    def test_safe_int_valid(self):
        from src.utils.visualizer import _safe_int

        self.assertEqual(_safe_int('42'), 42)
        self.assertEqual(_safe_int('3.7'), 3)

    def test_safe_int_invalid(self):
        from src.utils.visualizer import _safe_int

        self.assertIsNone(_safe_int(None))
        self.assertIsNone(_safe_int(''))
        self.assertIsNone(_safe_int('xyz'))


class LoadUltralyticsHistoryTests(unittest.TestCase):
    """Tests for load_ultralytics_history function."""

    def _write_results_csv(self, rows):
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8'
        )
        writer = csv.DictWriter(
            tmp, fieldnames=['epoch', 'train/loss', 'val/loss', 'metrics/accuracy_top1']
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp.close()
        return tmp.name

    def test_loads_valid_history(self):
        from src.utils.visualizer import load_ultralytics_history

        rows = [
            {
                'epoch': '1',
                'train/loss': '2.5',
                'val/loss': '2.3',
                'metrics/accuracy_top1': '0.65',
            },
            {
                'epoch': '2',
                'train/loss': '1.8',
                'val/loss': '1.7',
                'metrics/accuracy_top1': '0.78',
            },
        ]
        path = self._write_results_csv(rows)
        try:
            history = load_ultralytics_history(path)
            self.assertIsNotNone(history)
            self.assertEqual(len(history['epochs']), 2)
            self.assertEqual(len(history['train_loss']), 2)
            self.assertEqual(len(history['top1']), 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_file_returns_none(self):
        from src.utils.visualizer import load_ultralytics_history

        result = load_ultralytics_history('/nonexistent/path/results.csv')
        self.assertIsNone(result)

    def test_empty_csv_returns_none(self):
        from src.utils.visualizer import load_ultralytics_history

        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8'
        )
        writer = csv.DictWriter(tmp, fieldnames=['epoch'])
        writer.writeheader()
        tmp.close()

        try:
            result = load_ultralytics_history(tmp.name)
            self.assertIsNone(result)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_skips_empty_rows(self):
        from src.utils.visualizer import load_ultralytics_history

        rows = [
            {
                'epoch': '1',
                'train/loss': '2.5',
                'val/loss': '2.3',
                'metrics/accuracy_top1': '0.65',
            },
            {'epoch': '', 'train/loss': '', 'val/loss': '', 'metrics/accuracy_top1': ''},
        ]
        path = self._write_results_csv(rows)
        try:
            history = load_ultralytics_history(path)
            self.assertIsNotNone(history)
            self.assertEqual(len(history['epochs']), 1)
        finally:
            Path(path).unlink(missing_ok=True)


class PlotTrainingHistoryTests(unittest.TestCase):
    """Tests for plot_training_history function."""

    def test_saves_training_history_plot(self):
        from src.utils.visualizer import plot_training_history

        history = {
            'train_loss': [2.5, 1.8, 1.2],
            'val_loss': [2.3, 1.7, 1.3],
            'train_acc': [0.5, 0.7, 0.85],
            'val_acc': [0.55, 0.72, 0.8],
        }

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_training_history(history, save_path=path)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_history_does_nothing(self):
        from src.utils.visualizer import plot_training_history

        plot_training_history(None, save_path='/tmp/test_none.png')
        plot_training_history({}, save_path='/tmp/test_empty.png')


class PlotClassDistributionTests(unittest.TestCase):
    """Tests for plot_class_distribution function."""

    def test_saves_class_distribution_plot(self):
        from src.utils.visualizer import plot_class_distribution

        class_counts = {'crack': 500, 'no_crack': 500}

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_class_distribution(class_counts, save_path=path)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_class_counts(self):
        from src.utils.visualizer import plot_class_distribution

        plot_class_distribution(None, save_path='/tmp/test_none.png')
        plot_class_distribution({}, save_path='/tmp/test_empty.png')


class PlotComprehensiveReportTests(unittest.TestCase):
    """Tests for plot_comprehensive_report function."""

    def test_saves_comprehensive_report(self):
        from src.utils.visualizer import plot_comprehensive_report

        metrics = {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.91,
            'f1': 0.90,
            'roc_auc': 0.95,
            'confusion_matrix': [[45, 5], [8, 42]],
        }

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_comprehensive_report(metrics, save_path=path)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_with_run_metadata(self):
        from src.utils.visualizer import plot_comprehensive_report

        metrics = {'accuracy': 0.9, 'precision': 0.88, 'recall': 0.85, 'f1': 0.86, 'roc_auc': 0.92}
        metadata = {'run_name': 'test_run', 'epochs': 50, 'batch_size': 16}

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_comprehensive_report(metrics, save_path=path, run_metadata=metadata)
            self.assertTrue(Path(path).exists())
        finally:
            Path(path).unlink(missing_ok=True)


class PlotModelComparisonTests(unittest.TestCase):
    """Tests for plot_model_comparison function."""

    def test_saves_model_comparison(self):
        from src.utils.visualizer import plot_model_comparison

        models_data = [
            {'name': 'YOLOv8n', 'accuracy': 0.88, 'precision': 0.85, 'recall': 0.90, 'f1': 0.87},
            {'name': 'YOLOv8s', 'accuracy': 0.92, 'precision': 0.90, 'recall': 0.93, 'f1': 0.91},
        ]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_model_comparison(models_data, save_path=path)
            self.assertTrue(Path(path).exists())
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_models_data(self):
        from src.utils.visualizer import plot_model_comparison

        plot_model_comparison(None, save_path='/tmp/test_none.png')
        plot_model_comparison([], save_path='/tmp/test_empty.png')


if __name__ == '__main__':
    unittest.main()
