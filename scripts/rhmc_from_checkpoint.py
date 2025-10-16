#!/usr/bin/env python3
"""Deprecated RHMC sampling entry point.

Sampling runs can be triggered with:

    python scripts/run_workflow.py sampling --overrides settings.pipeline.outputs.metric_file=<metric.pt>
"""


def main() -> None:
    raise RuntimeError(
        "scripts/rhmc_from_checkpoint.py has been removed. "
        "Use scripts/run_workflow.py sampling with the appropriate overrides."
    )


if __name__ == "__main__":
    main()
