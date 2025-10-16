#!/usr/bin/env python3
"""
Unified workflow launcher.

Provides thin wrappers around ``run_experiment.py`` for the common workflows
that used to live in the legacy Hydra scripts (Stage A/B/C only runs, sweeps,
sampling, …). Each sub-command simply accumulates the appropriate
``settings.*`` overrides and delegates to ``run_experiment.py``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_run_experiment(
    overrides: Iterable[str],
    *,
    multirun: bool = False,
    dry_run: bool = False,
) -> None:
    """Invoke ``run_experiment.py`` with the provided Hydra overrides."""
    cmd: List[str] = [sys.executable, "run_experiment.py"]
    if multirun:
        cmd.append("-m")
    cmd.extend(overrides)
    pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
    print(f"👉 Executing: {pretty_cmd}")
    if dry_run:
        print("💤 Dry-run requested — command not executed.")
        return
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _extend_with_paths(overrides: List[str], **paths: str | None) -> None:
    """Append overrides for any non-None path keyword arguments."""
    for key, value in paths.items():
        if value:
            dotted_key = key.replace("__", ".")
            overrides.append(f"{dotted_key}={value}")


def _base_stage_overrides(
    *,
    run_a: bool,
    run_b: bool,
    run_c: bool,
    run_sampling: bool = False,
) -> List[str]:
    return [
        "settings.pipeline.mode=three_stage",
        f"settings.pipeline.run_stage_a={str(run_a).lower()}",
        f"settings.pipeline.run_stage_b={str(run_b).lower()}",
        f"settings.pipeline.run_stage_c={str(run_c).lower()}",
        f"settings.pipeline.run_sampling={str(run_sampling).lower()}",
    ]


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Unified workflow launcher for run_experiment.py")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def stage_parser(name: str, help_msg: str) -> argparse.ArgumentParser:
        p = subparsers.add_parser(name, help=help_msg)
        p.add_argument("--stage-a-dir", type=str, help="Override `settings.pipeline.outputs.stage_a_dir`.")
        p.add_argument("--stage-b-dir", type=str, help="Override `settings.pipeline.outputs.stage_b_dir`.")
        p.add_argument("--stage-c-dir", type=str, help="Override `settings.pipeline.outputs.stage_c_dir`.")
        p.add_argument("--metric-file", type=str, help="Override `settings.pipeline.outputs.metric_file`.")
        p.add_argument("--rhmc-samples", type=str, help="Override `settings.pipeline.outputs.rhmc_samples_file`.")
        p.add_argument("--output-base", type=str, help="Override `output.base_dir`.")
        p.add_argument("--experiment-name", type=str, help="Override `global.experiment_name`.")
        p.add_argument("--overrides", nargs="*", default=[], help="Additional Hydra overrides.")
        return p

    stage_parser("stage-a", "Run Stage A only.")
    stage_parser("stage-b", "Run Stage B only (expects existing Stage A artefacts).")
    stage_parser("stage-c", "Run Stage C only (expects Stage A/B artefacts).")

    pipeline = subparsers.add_parser("pipeline", help="Run the full three-stage pipeline.")
    pipeline.add_argument("--overrides", nargs="*", default=[], help="Additional Hydra overrides.")

    sampling = subparsers.add_parser("sampling", help="Run only the RHMC sampling stage.")
    sampling.add_argument("--overrides", nargs="*", default=[], help="Additional Hydra overrides.")

    sweep = subparsers.add_parser("sweep", help="Launch a Hydra multirun sweep.")
    sweep.add_argument("--overrides", nargs="*", default=[], help="Hydra overrides (axes, seeds, …).")

    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    dry_run = args.dry_run

    if remaining:
        print(f"⚠️ Unparsed extra arguments detected: {remaining}")
        print("   They will be appended to the Hydra overrides.")

    if args.command == "stage-a":
        overrides = _base_stage_overrides(run_a=True, run_b=False, run_c=False)
        overrides.extend([
            "settings.training.stage_overrides.stage_b.enabled=false",
            "settings.training.stage_overrides.stage_c.enabled=false",
        ])
        _extend_with_paths(
            overrides,
            output__base_dir=getattr(args, "output_base", None),
            global__experiment_name=getattr(args, "experiment_name", None),
            settings__pipeline__outputs__stage_a_dir=getattr(args, "stage_a_dir", None),
            settings__pipeline__outputs__stage_b_dir=getattr(args, "stage_b_dir", None),
            settings__pipeline__outputs__stage_c_dir=getattr(args, "stage_c_dir", None),
            settings__pipeline__outputs__metric_file=getattr(args, "metric_file", None),
            settings__pipeline__outputs__rhmc_samples_file=getattr(args, "rhmc_samples", None),
        )
        overrides.extend(args.overrides)
        overrides.extend(remaining)
        _run_run_experiment(overrides, dry_run=dry_run)
        return

    if args.command == "stage-b":
        overrides = _base_stage_overrides(run_a=False, run_b=True, run_c=False)
        overrides.extend([
            "settings.training.stage_overrides.stage_a.enabled=false",
            "settings.training.stage_overrides.stage_c.enabled=false",
        ])
        _extend_with_paths(
            overrides,
            output__base_dir=getattr(args, "output_base", None),
            global__experiment_name=getattr(args, "experiment_name", None),
            settings__pipeline__outputs__stage_a_dir=getattr(args, "stage_a_dir", None),
            settings__pipeline__outputs__stage_b_dir=getattr(args, "stage_b_dir", None),
            settings__pipeline__outputs__stage_c_dir=getattr(args, "stage_c_dir", None),
            settings__pipeline__outputs__metric_file=getattr(args, "metric_file", None),
            settings__pipeline__outputs__rhmc_samples_file=getattr(args, "rhmc_samples", None),
        )
        # Disable Stage A specific training tweaks
        overrides.append("settings.training.stage_overrides.stage_a.enabled=false")
        overrides.extend(args.overrides)
        overrides.extend(remaining)
        _run_run_experiment(overrides, dry_run=dry_run)
        return

    if args.command == "stage-c":
        overrides = _base_stage_overrides(run_a=False, run_b=False, run_c=True)
        overrides.extend([
            "settings.training.stage_overrides.stage_a.enabled=false",
            "settings.training.stage_overrides.stage_b.enabled=false",
        ])
        _extend_with_paths(
            overrides,
            output__base_dir=getattr(args, "output_base", None),
            global__experiment_name=getattr(args, "experiment_name", None),
            settings__pipeline__outputs__stage_a_dir=getattr(args, "stage_a_dir", None),
            settings__pipeline__outputs__stage_b_dir=getattr(args, "stage_b_dir", None),
            settings__pipeline__outputs__stage_c_dir=getattr(args, "stage_c_dir", None),
            settings__pipeline__outputs__metric_file=getattr(args, "metric_file", None),
            settings__pipeline__outputs__rhmc_samples_file=getattr(args, "rhmc_samples", None),
        )
        overrides.extend(args.overrides)
        overrides.extend(remaining)
        _run_run_experiment(overrides, dry_run=dry_run)
        return

    if args.command == "pipeline":
        overrides = ["settings.pipeline.mode=three_stage"]
        overrides.extend(args.overrides)
        overrides.extend(remaining)
        _run_run_experiment(overrides, dry_run=dry_run)
        return

    if args.command == "sampling":
        overrides = _base_stage_overrides(run_a=False, run_b=False, run_c=False, run_sampling=True)
        overrides.extend(args.overrides)
        overrides.extend(remaining)
        _run_run_experiment(overrides, dry_run=dry_run)
        return

    if args.command == "sweep":
        overrides = args.overrides + remaining
        if not overrides:
            raise ValueError("Provide at least one Hydra override axis for the sweep.")
        _run_run_experiment(overrides, multirun=True, dry_run=dry_run)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
