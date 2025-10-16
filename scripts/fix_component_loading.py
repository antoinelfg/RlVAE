#!/usr/bin/env python3
"""Component-loading helper retired.

Component loading diagnostics are now integrated in the main runner via
``EnhancedComponentLoader``. Extend that module or instrument
``run_experiment.py`` directly when additional debugging is required.
"""


def main() -> None:
    raise RuntimeError(
        "scripts/fix_component_loading.py has been retired. "
        "Use the integrated EnhancedComponentLoader or add ad-hoc diagnostics."
    )


if __name__ == "__main__":
    main()
