#!/usr/bin/env python3
"""Stage C diagnostic retired.

Recreate targeted probes using the staged workflows from
``scripts/run_workflow.py`` plus custom analysis notebooks.
"""


def main() -> None:
    raise RuntimeError(
        "scripts/diagnose_stagec_probe.py has been retired. "
        "Run staged experiments via scripts/run_workflow.py and analyse outputs manually."
    )


if __name__ == "__main__":
    main()
