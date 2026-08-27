import importlib.util
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "train_hash_regex_tail_guard_final", TOOLS / "train_hash_regex_tail_guard_final.py"
)
finalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = finalizer
assert SPEC.loader is not None
SPEC.loader.exec_module(finalizer)


class TailGuardFinalizerTest(unittest.TestCase):
    def test_finalizer_rejects_non_safe_candidate_report(self):
        with self.assertRaises(ValueError):
            finalizer.locked_configuration({"terminal_status": "safe-but-collapse"})


if __name__ == "__main__":
    unittest.main()
