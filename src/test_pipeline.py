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
from unittest.mock import patch
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
        import torch
        from build_index import extract_embedding

        class DummyModel(torch.nn.Module):
            def forward(self, tensor):
                return torch.ones((tensor.shape[0], 512))

        model = DummyModel()
        transform = lambda image: torch.zeros((3, 224, 224))

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


class TestMapillaryCollection(unittest.TestCase):
    def test_region_search_derives_center_and_radius_from_bbox(self):
        from download_data import region_search
        lat, lng, radius = region_search({"bbox": [-80.025, 40.425, -79.975, 40.455]})
        self.assertAlmostEqual(lat, 40.44)
        self.assertAlmostEqual(lng, -80.0)
        self.assertGreater(radius, 2_500)

    def test_candidate_query_uses_radius_not_bbox(self):
        import download_data as collector
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return {"data": []}
        with patch.object(collector.requests, "get", return_value=Response()) as request:
            self.assertEqual(collector.fetch_candidates("test-token", 40.44, -80.0, 3_500, 1), [])
        params = request.call_args.kwargs["params"]
        self.assertEqual((params["lat"], params["lng"], params["radius"]), (40.44, -80.0, 3_500))
        self.assertNotIn("bbox", params)

    def test_image_quality_filter(self):
        from download_data import quality_ok
        self.assertFalse(quality_ok(Image.new("RGB", (80, 80), "white")))
        self.assertFalse(quality_ok(Image.new("RGB", (224, 224), "white")))
        self.assertTrue(quality_ok(Image.effect_noise((224, 224), 80)))

    def test_perceptual_hash_detects_identical_images(self):
        from download_data import dhash
        image = Image.effect_noise((100, 100), 80)
        self.assertEqual(dhash(image), dhash(image.copy()))

    def test_limit_enforced_before_api_access(self):
        from download_data import download_dataset
        with self.assertRaises(ValueError):
            download_dataset(101)

    def test_download_writes_gps_and_provenance_metadata(self):
        import download_data as collector
        class Response:
            content = __import__("io").BytesIO()
            def raise_for_status(self):
                return None
        image_bytes = __import__("io").BytesIO()
        Image.effect_noise((224, 224), 80).save(image_bytes, format="JPEG")
        response = Response()
        response.content = image_bytes.getvalue()
        candidate = {"id": "street-123", "geometry": {"coordinates": [-79.99, 40.44]},
                     "thumb_1024_url": "https://example.invalid/thumb", "captured_at": 123,
                     "sequence": "sequence-1"}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(collector, "MAPILLARY_DIR", Path(directory)), \
             patch.object(collector, "get_token", return_value="test-token"), \
             patch.object(collector, "fetch_candidates", return_value=[candidate]), \
             patch.object(collector.requests, "get", return_value=response), \
             patch.object(collector.time, "sleep"):
            rows = collector.download_dataset(1)
            saved = json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
            saved_image_exists = Path(saved[0]["path"]).exists()
        self.assertEqual(len(rows), 1)
        self.assertEqual(saved[0]["image_id"], "street-123")
        self.assertAlmostEqual(saved[0]["lat"], 40.44)
        self.assertAlmostEqual(saved[0]["lon"], -79.99)
        self.assertEqual(saved[0]["source"], "Mapillary")
        self.assertTrue(saved_image_exists)


class TestPredictionAndEvaluation(unittest.TestCase):
    def test_predict_excludes_shape_mismatch_and_returns_top_k(self):
        import predict as module
        class Model:
            def __call__(self, tensor):
                import torch
                return torch.tensor([[1.0, 0.0]])
        class Torch:
            @staticmethod
            def inference_mode():
                import torch
                return torch.inference_mode()
        old_load_index, old_load_model = module.load_index, module.load_model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.jpg"
            Image.new("RGB", (20, 20), "red").save(path)
            records = [{"lat": 40+i, "lon": -80, "path": "", "region": "Pittsburgh"} for i in range(3)]
            module.load_index = lambda: (np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32), records)
            module.load_model = lambda: (Model(), lambda image: __import__("torch").zeros((3, 224, 224)), Torch)
            try:
                result = module.predict(path, top_k=2)
            finally:
                module.load_index, module.load_model = old_load_index, old_load_model
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["rank"], 1)

    def test_split_has_no_overlap_and_singleton_stays_reference(self):
        from evaluate import stratified_split
        rows = [{"region": "A", "filename": f"{i}"} for i in range(4)]
        rows.append({"region": "B", "filename": "singleton"})
        refs, queries = stratified_split(rows, 0.25)
        self.assertFalse({r["filename"] for r in refs} & {r["filename"] for r in queries})
        self.assertIn("singleton", {r["filename"] for r in refs})

    def test_index_builder_writes_embeddings_and_aligned_metadata(self):
        import build_index as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "raw.jpg"
            Image.new("RGB", (240, 160), "blue").save(image_path)
            record = {"image_id": "id-1", "path": str(image_path), "lat": 40.4,
                      "lon": -80.0, "region": "Pittsburgh", "source": "Mapillary"}
            patches = [patch.object(module, "INDEX_DIR", root / "index"),
                       patch.object(module, "PROCESSED_DIR", root / "processed"),
                       patch.object(module, "load_model", return_value=(None, None, None)),
                       patch.object(module, "extract_embedding", return_value=np.array([1, 0], dtype=np.float32))]
            with patches[0], patches[1], patches[2], patches[3], patch.object(module, "EMBEDDING_DIM", 2):
                stats = module.build_index([record])
            matrix = np.load(root / "index" / "embeddings.npy")
            indexed = json.loads((root / "index" / "metadata.json").read_text(encoding="utf-8"))
            processed_exists = Path(indexed[0]["path"]).exists()
        self.assertEqual(stats["image_count"], 1)
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(indexed[0]["image_id"], "id-1")
        self.assertTrue(processed_exists)


if __name__ == "__main__":
    unittest.main(verbosity=2)
