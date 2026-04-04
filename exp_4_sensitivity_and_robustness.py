from exp_common.experiments import build_arg_parser, run_exp_4_sensitivity_and_robustness


def main() -> None:
    parser = build_arg_parser("Run observation sensitivity and robustness ablations.")
    args = parser.parse_args()
    run_exp_4_sensitivity_and_robustness(
        output_root=args.output_root,
        device_name=args.device,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
