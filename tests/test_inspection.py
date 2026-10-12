from contextlib import contextmanager
from pathlib import Path
import unittest
from uuid import uuid4

import numpy as np

from inspection import auc, features, fit, image_score, infer, load_model, synthetic_image


@contextmanager
def fixture_directory():
    # Ordinary inherited permissions also work in restricted Windows workspaces.
    root = Path(__file__).resolve().parent / (".fixture-" + uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        for child in root.iterdir():
            child.unlink()
        root.rmdir()


class InspectionTests(unittest.TestCase):
    def test_auroc_and_ties(self):
        self.assertAlmostEqual(auc(np.array([0, 0, 1, 1]), np.array([0., 1., 2., 3.])), 1.0)
        self.assertAlmostEqual(auc(np.array([0, 1]), np.array([1., 1.])), 0.5)
        self.assertAlmostEqual(auc(np.array([0, 1]), np.array([2., 1.])), 0.0)

    def test_score_uses_top_two_patches(self):
        heat = np.array([[1., 2.], [3., 5.]])
        self.assertEqual(image_score(heat), 4.)

    def test_generator_reproducible_and_feature_dimensions(self):
        first = synthetic_image(np.random.default_rng(7), False)
        second = synthetic_image(np.random.default_rng(7), False)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(features(first).shape, (16, 16, 6))

    def test_fit_rejects_split_leakage(self):
        with fixture_directory() as directory:
            image = Path(directory) / "normal.png"
            synthetic_image(np.random.default_rng(1), False).save(image)
            with self.assertRaisesRegex(ValueError, "Identical"):
                fit([image] * 2, [image] * 5, Path(directory) / "model.npz")

    def test_model_roundtrip_and_defect_response(self):
        with fixture_directory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(13)
            files = []
            for i in range(35):
                path = root / f"{i}.png"
                synthetic_image(rng, False).save(path)
                files.append(path)
            model_path = root / "model.npz"
            fit(files[:24], files[24:34], model_path)
            model = load_model(model_path)
            normal_score, _, _ = infer(files[34], model)
            defect = root / "defect.png"
            synthetic_image(rng, True).save(defect)
            defect_score, _, _ = infer(defect, model)
            self.assertGreater(defect_score, normal_score)
            self.assertGreater(defect_score, float(model["threshold"]))


if __name__ == "__main__":
    unittest.main()
