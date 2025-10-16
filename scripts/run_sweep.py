#!/usr/bin/env python3
"""Deprecated sweep runner.

Hydra multiruns can be launched through ``scripts/run_workflow.py sweep``. For
example:

    python scripts/run_workflow.py sweep model.latent_dim='[8,16]' seed='range(3)'
"""


def main() -> None:
    raise RuntimeError(
        "scripts/run_sweep.py has been removed. "
        "Use `python scripts/run_workflow.py sweep ...` to launch multiruns."
    )


if __name__ == "__main__":
    main()
