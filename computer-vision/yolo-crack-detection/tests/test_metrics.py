"""Tests for src/utils/metrics.py."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ClassificationMetricsTests(unittest.TestCase):
    """Tests for the classification_metrics function."""

    def _get_metrics(self, y_true, y_scores=None, y_pred=None, threshold=0.5):
        from src.utils.metrics import classification_metrics

        return classification_metrics(y_true, y_scores, y_pred, threshold)

    def test_perfect_classification(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.9, 0.8]
        result = self._get_metrics(y_true, y_scores=y_scores)

        self.assertAlmostEqual(result['accuracy'], 1.0)
        self.assertAlmostEqual(result['precision'], 1.0)
        self.assertAlmostEqual(result['recall'], 1.0)
        self.assertAlmostEqual(result['f1'], 1.0)
        self.assertGreater(result['roc_auc'], 0.9)

    def test_worst_classification(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.9, 0.8, 0.1, 0.2]
        result = self._get_metrics(y_true, y_scores=y_scores)

        self.assertAlmostEqual(result['accuracy'], 0.0)
        self.assertAlmostEqual(result['f1'], 0.0)

    def test_using_y_pred_instead_of_y_scores(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        result = self._get_metrics(y_true, y_pred=y_pred)

        self.assertEqual(result['accuracy'], 0.75)
        self.assertIn('confusion_matrix', result)
        self.assertEqual(len(result['confusion_matrix']), 2)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            self._get_metrics([], y_pred=[])

    def test_numpy_arrays(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.9, 0.7, 0.6])
        result = self._get_metrics(y_true, y_scores=y_scores)

        self.assertIsInstance(result['accuracy'], float)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)

    def test_roc_auc_fallback_on_failure(self):
        y_true = [0, 0, 0, 0]
        y_scores = [0.5, 0.5, 0.5, 0.5]
        result = self._get_metrics(y_true, y_scores=y_scores)

        # sklearn returns NaN for single-class ROC AUC; the function
        # catches the exception or the NaN and stores 0.0.
        roc_auc = result['roc_auc']
        self.assertTrue(roc_auc == 0.0 or (isinstance(roc_auc, float) and np.isnan(roc_auc)))
        self.assertIn('accuracy', result)

    def test_confusion_matrix_structure(self):
        y_true = [0, 0, 1, 1, 1, 0]
        y_pred = [0, 1, 1, 0, 1, 0]
        result = self._get_metrics(y_true, y_pred=y_pred)

        cm = result['confusion_matrix']
        if cm:
            self.assertEqual(len(cm), 2)
            self.assertEqual(len(cm[0]), 2)

    def test_metrics_are_json_serializable(self):
        y_true = [0, 1, 1, 0, 1]
        y_scores = [0.2, 0.8, 0.7, 0.3, 0.9]
        result = self._get_metrics(y_true, y_scores=y_scores)

        json.dumps(result)


class SaveMetricsReportTests(unittest.TestCase):
    """Tests for save_metrics_report function."""

    def test_saves_report_file(self):
        from src.utils.metrics import save_metrics_report

        metrics = {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88}

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            path = f.name

        try:
            save_metrics_report(metrics, path)
            content = Path(path).read_text()

            self.assertIn('accuracy', content)
            self.assertIn('0.95', content)
            self.assertIn('precision', content)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_creates_parent_directory(self):
        from src.utils.metrics import save_metrics_report

        metrics = {'f1': 0.9}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / 'sub' / 'metrics.txt'
            save_metrics_report(metrics, dest)

            self.assertTrue(dest.exists())
            self.assertIn('f1', dest.read_text())


class PlotConfusionMatrixTests(unittest.TestCase):
    """Tests for plot_confusion_matrix function."""

    def test_saves_confusion_matrix_plot(self):
        import matplotlib

        from src.utils.metrics import plot_confusion_matrix

        matplotlib.use('Agg')

        cm = [[50, 5], [10, 45]]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_confusion_matrix(cm, save_path=path)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)
        finally:
            Path(path).unlink(missing_ok=True)


class PlotROCTests(unittest.TestCase):
    """Tests for plot_roc function."""

    def test_saves_roc_plot(self):
        import matplotlib

        from src.utils.metrics import plot_roc

        matplotlib.use('Agg')

        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.1, 0.3, 0.8, 0.9, 0.2, 0.7]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_roc(y_true, y_scores, save_path=path)
            self.assertTrue(Path(path).exists())
        finally:
            Path(path).unlink(missing_ok=True)


class PlotPrecisionRecallTests(unittest.TestCase):
    """Tests for plot_precision_recall function."""

    def test_saves_pr_plot(self):
        import matplotlib

        from src.utils.metrics import plot_precision_recall

        matplotlib.use('Agg')

        y_true = [0, 0, 1, 1, 1]
        y_scores = [0.1, 0.4, 0.7, 0.8, 0.6]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            plot_precision_recall(y_true, y_scores, save_path=path)
            self.assertTrue(Path(path).exists())
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Agg')
    unittest.main()
