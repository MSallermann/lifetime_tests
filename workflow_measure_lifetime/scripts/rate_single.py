from hopfion_lifetimes.calculation_setup.parameters import HamiltonianParameters
from hopfion_lifetimes.scripts import rate_llg
from spirit import system
from pathlib import Path


def main(
    hamiltonian: HamiltonianParameters,
    params: rate_llg.MethodParameters,
    sz: float,
):

    def has_switched(p_state, params):
        spins = system.get_spin_directions(p_state)
        return spins[0][2] > sz

    rate_llg.run(hamiltonian=hamiltonian, params=params, has_switched=has_switched)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some data.")

    parser.add_argument(
        "--system",
        required=True,
        type=Path,
        help="Path to the hamiltonian parameter_file",
    )

    parser.add_argument(
        "--params",
        type=Path,
        required=False,
        help="Path to the method parameter file",
    )

    parser.add_argument(
        "--n_events",
        type=int,
        required=True,
        help="number of events",
    )

    parser.add_argument(
        "-i",
        "--image",
        type=Path,
        required=True,
        help="Path to the initial image",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the trajectory folder",
    )

    parser.add_argument(
        "--sz",
        required=False,
        default=0.95,
        type=float,
        help="Critical sz value",
    )

    parser.add_argument(
        "--temp",
        default=None,
        required=False,
        type=float,
        help="temperature in Kelvin",
    )

    parser.add_argument(
        "--damping",
        default=None,
        required=False,
        type=float,
        help="damping",
    )

    parser.add_argument(
        "--field_x",
        default=None,
        required=False,
        type=float,
        help="Magnetic field in [1,0,0] (x) direction in Tesla",
    )

    # Parse the arguments
    args = parser.parse_args()

    parameters_hamiltonian = HamiltonianParameters.from_file(args.system)
    if not args.field_x is None:
        parameters_hamiltonian.ext_field = [args.field_x, 0.0, 0.0]

    cli_update_dict = dict(
        trajectory_folder=args.output,
        path_initial_image=args.image,
        n_events=args.n_events,
    )
    if not args.damping is None:
        cli_update_dict["damping"] = args.damping

    if not args.temp is None:
        cli_update_dict["temperature"] = args.temp

    if not args.params is None:
        parameters_method = rate_llg.MethodParameters.from_file(
            args.params, cli_update_dict
        )
    else:
        print(cli_update_dict)
        parameters_method = rate_llg.MethodParameters(**cli_update_dict)

    rate_llg.MethodParameters.model_validate(parameters_method)
    HamiltonianParameters.model_validate(parameters_hamiltonian)

    output_file = args.output

    main(parameters_hamiltonian, parameters_method, args.sz)
