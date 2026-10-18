"""Roundtrip a self-constructed wrapper, never deserialize real weights here."""
import io
import unittest
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from experiments.english_v1.stat_style_refine import StyleOnlyDecisionHead

class SerializedCompatibilityChecks(unittest.TestCase):
    def test_original_module_class_roundtrip_and_column_projection(self):
        estimator = LinearSVC()
        estimator.classes_ = np.array([0, 1])
        estimator.coef_ = np.array([[2.0, -1.0]])
        estimator.intercept_ = np.array([0.25])
        estimator.n_features_in_ = 2
        wrapper = StyleOnlyDecisionHead(estimator, [2, 0])
        payload = io.BytesIO()
        joblib.dump(wrapper, payload)
        payload.seek(0)
        restored = joblib.load(payload)
        self.assertEqual(type(restored).__module__, "experiments.english_v1.stat_style_refine")
        self.assertTrue(np.array_equal(restored.classes_, [0, 1]))
        values = np.array([[1., 9., 3.], [2., 8., 4.]])
        np.testing.assert_allclose(restored.decision_function(values), [5.25, 6.25])

if __name__ == "__main__":
    unittest.main()
