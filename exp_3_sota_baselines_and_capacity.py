from exp_common.experiments import build_arg_parser, run_exp_3_sota_baselines_and_capacity


def main() -> None:
    parser = build_arg_parser("Run baseline capacity and Pareto frontier experiments.")
    args = parser.parse_args()
    run_exp_3_sota_baselines_and_capacity(
        output_root=args.output_root,
        device_name=args.device,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
