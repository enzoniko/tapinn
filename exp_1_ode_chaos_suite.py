from exp_common.experiments import build_arg_parser, run_exp_1_ode_chaos_suite


def main() -> None:
    parser = build_arg_parser("Benchmark TAPINN across chaotic ODE systems.")
    args = parser.parse_args()
    run_exp_1_ode_chaos_suite(
        output_root=args.output_root,
        device_name=args.device,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
