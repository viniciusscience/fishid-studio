import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

for mod in ("torch", "cv2", "ultralytics", "PIL", "PIL.Image"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from monitoramento_em import _classify_crop_via_fishid_api


class TestPredictIntegration(unittest.TestCase):
    def test_reads_root_fields_from_predict_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            crop = Path(tmp) / "crop.jpg"
            crop.write_bytes(b"fake-jpeg")

            payload = {
                "display_name": "Sardinella brasiliensis",
                "top_species": "Sardinha-verdadeira",
                "scientific_name": "Sardinella brasiliensis",
                "confidence": 0.91234,
            }
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.read.return_value = json.dumps(payload).encode("utf-8")

            with patch("urllib.request.urlopen", return_value=mock_resp):
                res = _classify_crop_via_fishid_api("http://localhost:8000", crop)

            self.assertEqual(res, ("Sardinella brasiliensis", 0.9123))

    def test_prints_warn_when_api_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            crop = Path(tmp) / "crop.jpg"
            crop.write_bytes(b"fake-jpeg")

            buf = io.StringIO()
            with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("offline")):
                with redirect_stdout(buf):
                    res = _classify_crop_via_fishid_api("http://localhost:8000", crop)

            self.assertIsNone(res)
            self.assertIn("[WARN]", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
