"""Run the final manuscript analyses with prepared local inputs."""

from pathlib import Path
import argparse, os, subprocess, sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["study1", "study2", "study3", "all"])
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also run the supplementary Study 1 and Study 2 reference analyses.",
    )
    p.add_argument(
        "--coverage-inputs",
        type=Path,
        help="Optional caption-available coverage subset, retaining the same final vectors.",
    )
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        env.setdefault(key, "2")
    for key, folder in [
        ("study1", "03_study_1"),
        ("study2", "04_study_2"),
        ("study3", "05_study_3"),
    ]:
        if a.stage not in [key, "all"]:
            continue
        args = [
            sys.executable,
            str(root / folder / "run.py"),
            "--inputs",
            str(a.inputs.resolve()),
            "--outputs",
            str((a.outputs / key).resolve()),
        ]
        if key == "study2":
            if a.diagnostics:
                args += ["--diagnostics"]
            if a.coverage_inputs:
                args += ["--coverage-inputs", str(a.coverage_inputs.resolve())]
        subprocess.run(args, env=env, check=True)
        if key == "study1" and a.diagnostics:
            subprocess.run(
                [
                    sys.executable,
                    str(root / folder / "reordering.py"),
                    "--inputs",
                    str(a.inputs.resolve()),
                    "--outputs",
                    str((a.outputs / key).resolve()),
                ],
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
