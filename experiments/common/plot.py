"""Render paper-result figures from the generated CSV tables."""

import argparse
import csv
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("04", "05", "06", "07"), required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    tables = args.tables_dir
    out = args.output_dir or tables
    out.mkdir(parents=True, exist_ok=True)
    scales = (50, 100, 300, 1000, 2900)
    if args.section == "04":
        rows, fits = read(tables/"scaling_seed_means.csv"), read(tables/"scaling_power_laws.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for n in range(1, 6):
            group = sorted([r for r in rows if int(r["N"]) == n], key=lambda r: int(r["D"]))
            x = np.array([int(r["D"]) for r in group])
            line, = axes[0].loglog(x, [float(r["rmse"]) for r in group], "o-", label=f"N{n}")
            fit = next(r for r in fits if int(r["N"]) == n)
            axes[0].plot(x, float(fit["a"])*x**(-float(fit["alpha"])), "--", color=line.get_color())
        for d in scales:
            group = sorted([r for r in rows if int(r["D"]) == d], key=lambda r: int(r["N"]))
            axes[1].loglog([float(r["parameters_m"]) for r in group], [float(r["rmse"]) for r in group], "o-", label=f"D{d}")
        axes[0].set_xlabel("Demonstrations per task")
        axes[1].set_xlabel("Trainable parameters (millions)")
        for ax in axes:
            ax.set_ylabel("Holdout action RMSE"); ax.legend(frameon=False); ax.grid(alpha=.2)
        name = "scaling"
    elif args.section == "05":
        rows = read(tables/"token_interface.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for k in (1, 4, 50):
            group = sorted([r for r in rows if int(r["K"]) == k and int(r["seed"]) == 0], key=lambda r: int(r["D"]))
            for ax, key in zip(axes, ("sr_percent", "rmse")):
                ax.semilogx([int(r["D"]) for r in group], [float(r[key]) for r in group], "o-", label=f"K={k}")
        axes[0].set_ylabel("Success rate (%)"); axes[1].set_ylabel("Holdout action RMSE")
        for ax in axes:
            ax.set_xlabel("Demonstrations per task"); ax.legend(frameon=False); ax.grid(alpha=.2)
        name = "token_interface"
    elif args.section == "06":
        rows = read(tables/"history_truncation.csv")
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
        limit = max(abs(float(r["delta_sr_pp"])) for r in rows)
        values_by_cell = {(int(r["seed"]), int(r["C_test"]), int(r["N"]), int(r["D"])): float(r["delta_sr_pp"]) for r in rows}
        for seed in (0, 1):
            for j, c in enumerate((1, 2, 5)):
                values = np.array([[values_by_cell[seed, c, n, d] for d in scales] for n in range(1, 6)])
                ax = axes[seed, j]
                im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
                for y in range(5):
                    for x in range(5):
                        ax.text(x, y, f"{values[y,x]:+.1f}", ha="center", va="center", fontsize=8)
                ax.set_xticks(range(5), [str(d) for d in scales])
                ax.set_yticks(range(5), [f"N{n}" for n in range(1, 6)])
                ax.set_title(f"Seed {seed}, C_test={c}")
                ax.set_xlabel("Demonstrations per task")
        fig.colorbar(im, ax=axes, label="SR change from C=10 (percentage points)", shrink=.75)
        name = "history_truncation"
    else:
        rows, stress = read(tables/"ranking_success.csv"), read(tables/"ranking_stress.csv")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
        axes[0].bar([r["C_test"] for r in rows], [float(r["sr_percent"]) for r in rows], color="#3e8f86")
        axes[0].set(xlabel="Visible history (policy timesteps)", ylabel="Evaluator success (%)", ylim=(0, 100))
        axes[1].bar([r["seed"] for r in stress], [float(r["last_correct_swap_seconds"]) for r in stress], color="#3b84a5")
        axes[1].axhline(608*.24, ls="--", color="black", label="History window starts sliding")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set(xlabel="Environment seed", ylabel="Last correct swap (simulated seconds)")
        axes[1].legend(frameon=False, fontsize=8)
        name = "ranking"
    for suffix in ("pdf", "svg"):
        fig.savefig(out/f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
