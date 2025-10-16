#!/usr/bin/env python3
"""Initialization fix helper retired.

Any bespoke fixes should be implemented directly inside the modern training
pipeline or bespoke notebooks.
"""


def main() -> None:
    raise RuntimeError(
        "scripts/fix_multiple_init_phases.py has been retired. "
        "Integrate any required adjustments into the current codebase instead."
    )


if __name__ == "__main__":
    main()
