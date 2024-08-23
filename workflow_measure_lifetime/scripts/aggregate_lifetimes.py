from pathlib import Path
from typing import List

from hopfion_lifetimes.calculation_setup.parameters import HamiltonianParameters
from hopfion_lifetimes.scripts.compute_determinant import DeterminantResult

import lifetime_from_mfpt

import pandas as pd


def main(
    temperatures: List[float],
    mfpt_results: List[lifetime_from_mfpt.MFPTLifetimeResult],
    output: Path,
):

    data = dict(
        temperatures=temperatures,
        mfpt_results=[r.model_dump_json() for r in mfpt_results],
    )

    df = pd.DataFrame(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        required=True,
        help="List of temperatures",
    )

    parser.add_argument(
        "--mfpts",
        type=Path,
        nargs="+",
        required=True,
        help="List of paths to the mfpt result files",
    )

    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="The output file"
    )

    args = parser.parse_args()

    mfpt = [lifetime_from_mfpt.MFPTLifetimeResult.from_file(f) for f in args.mfpts]

    main(args.temps, mfpt, args.output)
