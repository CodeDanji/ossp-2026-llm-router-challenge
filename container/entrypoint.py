# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Container entry point for the frozen PromptBudget v3.3 router."""

from __future__ import annotations

import sys
from pathlib import Path


def _runtime_root(entrypoint_path: Path) -> Path:
    """Find the repository root before or after Docker copies the entrypoint."""

    parent = entrypoint_path.parent
    return parent.parent if parent.name == "container" else parent


_ROOT = _runtime_root(Path(__file__).resolve())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.hash_regex import main as hash_regex_main


_ARTIFACT = _ROOT / "build" / "hash-regex-tail-guard" / "final-artifact.json"


def main(argv: list[str] | None = None) -> int:
    """Run the frozen hash-regex router with its bundled v3.3 artifact."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return hash_regex_main([*arguments, "--artifact", str(_ARTIFACT)])


if __name__ == "__main__":
    raise SystemExit(main())
