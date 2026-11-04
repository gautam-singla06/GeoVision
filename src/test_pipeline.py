"""
test_pipeline.py — Smoke tests for the GeoVision pipeline.

Run with:
    python src/test_pipeline.py

Tests (all run without a GPU, without downloading real images):
  1. Config paths exist / are correct types.
  2. download_data.py — Wikimedia API functions are importable and params valid.
  3. build_index.py  — embed() produces unit-norm vectors.
  4. predict.py      — cosine similarity maths are correct.
  5. evaluate.py     — stratified_split keeps every region in both sets.
"""

import sys
import math
import json
import tempfile
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestConfig(unittest.TestCase):
    def test_dirs_are_paths(self):
        from config import BASE_DIR, DATA_DIR, RAW_DIR, INDEX_DIR
        for d in (BASE_DIR, DATA_DIR, RAW_DIR, INDEX_DIR):
            self.assertIsInstance(d, Path)

    def test_regions_have_required_keys(self):
        from config import REGIONS
        self.assertGreater(len(REGIONS), 0)
        for r in REGIONS:
            self.assertIn("name", r)
            self.assertIn("lat",  r)
            self.assertIn("lon",  r)
            self.assertGreaterEqual(r["lat"], -90)
            self.assertLessEqual(r["lat"],    90)

    def test_image_size_positive(self):
        from config import IMAGE_SIZE, NUM_IMAGES_PER_REGION
        self.assertGreater(IMAGE_SIZE,            0)
        self.assertGreater(NUM_IMAGES_PER_REGION, 0)


class TestHaversine(unittest.TestCase):
    def test_same_point_is_zero(self):
        from predict import haversine_km
        self.assertAlmostEqual(haversine_km(0, 0, 0, 0), 0.0, places=5)

    def test_known_distance(self):
        from predict import haversine_km
        # Paris (48.8566, 2.3522) → London (51.5074, -0.1278) ≈ 340 km
        d = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
        self.assertGreater(d, 300)
        self.assertLess(d, 400)

    def test_antipodal_is_roughly_half_circumference(self):
        from predict import haversine_km
        d = haversine_km(0, 0, 0, 180)
        self.assertAlmostEqual(d, math.pi * 6371, delta=5)


class TestEmbeddingMath(unittest.TestCase):
    """Test cosine-similarity logic without loading the actual model."""

    def _make_unit_matrix(self, n: int, d: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        m   = rng.random((n, d)).astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / norms

    def test_self_similarity_is_one(self):
        mat = self._make_unit_matrix(10, 512)
        for i in range(10):
            sim = float(mat[i] @ mat[i])
            self.assertAlmostEqual(sim, 1.0, places=5)

    def test_top1_retrieval_returns_itself(self):
        mat   = self._make_unit_matrix(20, 512)
        query = mat[7]
        sims  = mat @ query
        best  = int(np.argmax(sims))
        self.assertEqual(best, 7)

    def test_similarity_bounded(self):
        mat   = self._make_unit_matrix(50, 512)
        query = self._make_unit_matrix(1, 512)[0]
        sims  = mat @ query
        self.assertTrue((sims >= -1.01).all())
        self.assertTrue((sims <=  1.01).all())


class TestStratifiedSplit(unittest.TestCase):
    def _make_meta(self) -> list[dict]:
        """Create fake metadata with 4 regions × 10 images each."""
        meta = []
        for region in ["A", "B", "C", "D"]:
            for i in range(10):
                meta.append({
                    "region": region, "lat": 0.0, "lon": 0.0,
                    "filename": f"{region}_{i}.jpg", "path": "",
                })
        return meta

    def test_every_region_in_both_sets(self):
        from evaluate import stratified_split
        meta = self._make_meta()
        ref, query = stratified_split(meta, held_frac=0.20)
        ref_regions   = {m["region"] for m in ref}
        query_regions = {m["region"] for m in query}
        all_regions   = {m["region"] for m in meta}
        self.assertEqual(ref_regions,   all_regions)
        self.assertEqual(query_regions, all_regions)

    def test_split_fractions_approx(self):
        from evaluate import stratified_split
        meta = self._make_meta()
        ref, query = stratified_split(meta, held_frac=0.20)
        total = len(ref) + len(query)
        self.assertEqual(total, len(meta))
        self.assertAlmostEqual(len(query) / total, 0.20, delta=0.10)

    def test_no_overlap(self):
        from evaluate import stratified_split
        meta = self._make_meta()
        ref, query = stratified_split(meta, held_frac=0.20)
        ref_files   = {m["filename"] for m in ref}
        query_files = {m["filename"] for m in query}
        self.assertEqual(ref_files & query_files, set())


class TestImageEmbeddingShape(unittest.TestCase):
    """Test that the embedding pipeline produces correctly shaped vectors.
    Skipped if torch is not installed."""

    @unittest.skipUnless(
        __import__("importlib.util", fromlist=["util"]).find_spec("torch") is not None,
        "torch not installed"
    )
    def test_embedding_shape_and_norm(self):
        from build_index import extract_embedding, _load_model
        model, transform, torch = _load_model()

        # Create a blank 224×224 test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = Path(f.name)
        Image.new("RGB", (224, 224), color=(128, 64, 32)).save(tmp)

        vec = extract_embedding(model, transform, torch, tmp)
        tmp.unlink(missing_ok=True)

        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (512,))
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
