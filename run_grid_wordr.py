import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default="results/grid_wordr")
    parser.add_argument("--run-script", default="run_experiment_wordr.py")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k-list", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--threshold-list", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.top_k_list) * len(args.threshold_list)
    run_idx = 0

    for top_k in args.top_k_list:
        for threshold in args.threshold_list:
            run_idx += 1

            tag = f"topk{top_k}_thr{str(threshold).replace('.', 'p')}"
            output_path = out_dir / f"{args.dataset}_{tag}.jsonl"

            cmd = [
                "python",
                args.run_script,
                "--input", args.input,
                "--dataset", args.dataset,
                "--output", str(output_path),
                "--top-k", str(top_k),
                "--conf-threshold", str(threshold),
            ]

            if args.max_samples is not None:
                cmd += ["--max-samples", str(args.max_samples)]

            if args.overwrite:
                cmd += ["--overwrite"]

            print("\n" + "=" * 80)
            print(f"[{run_idx}/{total}] Running {tag}")
            print("Output:", output_path)
            print("Command:", " ".join(cmd))
            print("=" * 80)

            subprocess.run(cmd, check=True)
    print("\nAll grid runs finished.")


if __name__ == "__main__":
    main()