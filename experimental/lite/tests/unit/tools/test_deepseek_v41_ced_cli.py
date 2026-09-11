"""The CED command must execute validation and propagate failures."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / "tools/validate_deepseek_v41_ced.py"


class CedCliTest(unittest.TestCase):
    def test_model_argument_is_required(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--official-model", result.stderr)

    def test_invalid_official_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.py"
            model.write_text("# No official methods: validation must fail.\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--official-model", str(model)],
                capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("ok:", result.stdout)


if __name__ == "__main__":
    unittest.main()
