from exp_common.experiments import build_arg_parser, run_exp_5_theoretical_optimization_landscape


def main() -> None:
    parser = build_arg_parser("Analyze the optimization landscape and NTK evolution.")
    args = parser.parse_args()
    run_exp_5_theoretical_optimization_landscape(
        output_root=args.output_root,
        device_name=args.device,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
