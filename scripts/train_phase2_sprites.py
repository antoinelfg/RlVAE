#!/usr/bin/env python3
"""Deprecated Stage B training entry point.

Use:

    python scripts/run_workflow.py stage-b --stage-a-dir <path-to-stage-a>
"""


def main() -> None:
    raise RuntimeError(
        "scripts/train_phase2_sprites.py has been removed. "
        "Use scripts/run_workflow.py stage-b (optionally pointing at Stage A artefacts)."
    )


if __name__ == "__main__":
    main()
