#!/usr/bin/env python3
"""Deprecated Stage A training entry point.

Use:

    python scripts/run_workflow.py stage-a
"""


def main() -> None:
    raise RuntimeError(
        "scripts/train_phase1_sprites.py has been removed. "
        "Run `python scripts/run_workflow.py stage-a` with optional overrides instead."
    )


if __name__ == "__main__":
    main()
