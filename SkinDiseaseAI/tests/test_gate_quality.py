"""Quick tests that need no dataset and no TensorFlow:  python -m unittest discover -s tests -t ."""
import io, unittest
import numpy as np
from PIL import Image, ImageFilter
from src.gate import check_image, quality_features, quality_issues, read_image

LIMITS = {"lum": 60.0, "contrast": 15.0, "blur": 40.0, "dark_frac": 0.5, "sat_frac": 0.2}

def textured(seed=0, size=300):
    r = np.random.default_rng(seed)
    base = Image.fromarray((r.random((60, 60, 3)) * 160 + 60).astype("uint8")).resize((size, size), Image.BICUBIC)
    noise = (r.random((size, size, 3)) * 40).astype("uint8")
    return Image.fromarray(np.clip(np.asarray(base, dtype=np.int16) + noise - 20, 0, 255).astype("uint8"))

def to_bytes(im, fmt="JPEG"):
    b = io.BytesIO(); im.save(b, fmt); return b.getvalue()

class FileChecks(unittest.TestCase):
    def test_empty(self): self.assertIsNotNone(read_image(b"")[1])
    def test_corrupt(self): self.assertIsNotNone(read_image(b"not an image at all")[1])
    def test_truncated_jpeg(self): self.assertIsNotNone(read_image(to_bytes(textured())[:200])[1])
    def test_tiny(self): self.assertIsNotNone(read_image(to_bytes(textured(size=30)))[1])
    def test_unsupported_gif(self): self.assertIsNotNone(read_image(to_bytes(textured().convert("P"), "GIF"))[1])
    def test_valid_jpeg_png(self):
        for fmt in ("JPEG", "PNG"):
            img, err = read_image(to_bytes(textured(), fmt)); self.assertIsNone(err); self.assertIsNotNone(img)
    def test_check_image_never_raises_on_corrupt(self):
        v = check_image(b"garbage"); self.assertEqual(v["status"], "rejected")

class Quality(unittest.TestCase):
    def test_good_image_has_no_issues(self):
        self.assertEqual(quality_issues(quality_features(textured()), LIMITS), [])
    def test_dark_image(self):
        dark = Image.fromarray((np.asarray(textured(), dtype=np.float32) * 0.08).astype("uint8"))
        self.assertIn("lum", quality_issues(quality_features(dark), LIMITS))
    def test_mostly_black_artwork(self):
        a = np.zeros((300, 300, 3), dtype="uint8"); a[100:130, 100:130] = 200
        self.assertIn("dark_frac", quality_issues(quality_features(Image.fromarray(a)), LIMITS))
    def test_blurred_image(self):
        sharp = quality_features(textured()); blur = quality_features(textured().filter(ImageFilter.GaussianBlur(12)))
        self.assertLess(blur["blur"], sharp["blur"] / 5)
        self.assertIn("blur", quality_issues(blur, LIMITS))
    def test_glare(self):
        im = textured(); a = np.asarray(im).copy(); a[:, :150] = 255
        self.assertIn("sat_frac", quality_issues(quality_features(Image.fromarray(a)), LIMITS))
    def test_flat_low_detail_image(self):
        flat = Image.new("RGB", (300, 300), (120, 120, 120))
        self.assertIn("contrast", quality_issues(quality_features(flat), LIMITS))

if __name__ == "__main__":
    unittest.main()
