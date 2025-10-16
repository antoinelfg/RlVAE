#!/usr/bin/env python3
"""Deprecated three-stage orchestrator.

Use ``scripts/run_workflow.py`` for staged executions, e.g.:

    python scripts/run_workflow.py pipeline
    python scripts/run_workflow.py stage-a
    python scripts/run_workflow.py stage-b
"""


def main() -> None:
    raise RuntimeError(
        "scripts/orchestrate_three_stage.py has been removed. "
        "Use scripts/run_workflow.py for staged executions with the unified settings."
    )


if __name__ == "__main__":
    main()
