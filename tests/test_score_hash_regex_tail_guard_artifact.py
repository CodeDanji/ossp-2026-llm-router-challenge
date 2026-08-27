import importlib.util
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "score_hash_regex_tail_guard_artifact", TOOLS / "score_hash_regex_tail_guard_artifact.py"
)
scorer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scorer
assert SPEC.loader is not None
SPEC.loader.exec_module(scorer)


class TailGuardArtifactScorerTest(unittest.TestCase):
    def test_rejects_non_dev_input(self):
        with self.assertRaises(ValueError):
            scorer.require_dev_path(Path("data/materialized/train/inputs.json"))


if __name__ == "__main__":
    unittest.main()
