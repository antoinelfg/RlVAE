#!/usr/bin/env python3
"""Deprecated pipeline entry point.

The adaptive three-stage pipeline now lives behind ``scripts/run_workflow.py``.
Example:

    python scripts/run_workflow.py pipeline

Use the ``stage-a`` / ``stage-b`` / ``stage-c`` subcommands for partial runs.
"""


def main() -> None:
    raise RuntimeError(
        "scripts/adaptive_global_rlvae_pipeline.py has been removed. "
        "Use `python scripts/run_workflow.py pipeline` (or the stage-specific "
        "subcommands) with the unified configuration."
    )


if __name__ == "__main__":
    main()
