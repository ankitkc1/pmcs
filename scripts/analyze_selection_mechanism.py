
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODALITIES = ("FLAIR", "T1ce", "T1", "T2")
RUN_GRID = (
    ("A", 42, 3),
    ("A", 123, 4),
    ("A", 2026, 5),
    ("B", 42, 9),
    ("B", 123, 10),
    ("B", 2026, 11),
)
POOL_SIZES = {
    "A": (7, 2, 5, 4),
    "B": (2, 7, 5, 4),
}
RECURSION_SKIP = {
    "per_case_records",
    "per_case_stats",
    "case_results",
    "clients_dict",
    "clients_optim_dict",
    "global_encoders",
    "state_dict",
}
VALUES_ONLY = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--array-job", type=int, default=31922)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--mb-per-round", type=float, default=374.0)
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="write/print numerical tables without saving any figures",
    )
    parser.add_argument(
        "--out-dir",
        default="results/selection_mechanism_splitA_splitB_K2",
    )
    return parser.parse_args()


def locate_result(root, split, seed, task, array_job, k):
    exact = root / (
        "pmcs_pp_split{}_K{}_150r_seed{}_data20260905_job{}_task{}".format(
            split, k, seed, array_job, task
        )
    )
    if (exact / "metrics.json").is_file():
        return exact

    patterns = (
        "*split{}*K{}*seed{}*job{}*task{}*".format(
            split, k, seed, array_job, task
        ),
        "*split{}*K{}*seed{}*".format(split, k, seed),
    )
    for pattern in patterns:
        matches = sorted(
            path for path in root.glob(pattern) if (path / "metrics.json").is_file()
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Ambiguous result for Split {}, seed {}:\n{}".format(
                    split, seed, "\n".join(str(path) for path in matches)
                )
            )
    raise FileNotFoundError(
        "No result found for Split {}, K={}, seed={}, task={}".format(
            split, k, seed, task
        )
    )


def find_key(obj, target, path="$", found=None):
    if found is None:
        found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = "{}.{}".format(path, key)
            if key == target:
                found.append((child, value))
            if key not in RECURSION_SKIP:
                find_key(value, target, child, found)
    elif isinstance(obj, list) and len(obj) <= 200:
        for index, value in enumerate(obj):
            if isinstance(value, dict):
                find_key(value, target, "{}[{}]".format(path, index), found)
    return found


def numeric_round_items(per_round):
    if isinstance(per_round, list):
        return [(index, value) for index, value in enumerate(per_round)]
    if isinstance(per_round, dict):
        def order(item):
            text = str(item)
            return (0, int(text)) if text.isdigit() else (1, text)

        return [(key, per_round[key]) for key in sorted(per_round, key=order)]
    return []


def choose_coverage(metrics):
    candidates = find_key(metrics, "encoder_coverage")
    if not candidates:
        raise KeyError("encoder_coverage not found")

    def length(candidate):
        block = candidate[1]
        if not isinstance(block, dict):
            return -1
        return len(numeric_round_items(block.get("per_round")))

    path, coverage = max(candidates, key=length)
    if not isinstance(coverage, dict) or not coverage.get("per_round"):
        raise ValueError("{} has no encoder_coverage.per_round entries".format(path))
    return path, coverage


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def modality_value(block, modality, index):
    if isinstance(block, (list, tuple)) and len(block) >= len(MODALITIES):
        return block[index]
    if isinstance(block, dict):
        wanted = normalized(modality)
        for key, value in block.items():
            if normalized(key) == wanted:
                return value
        for key in (index, str(index)):
            if key in block:
                return block[key]
    return None


def scalar_from_dict(block, names):
    if not isinstance(block, dict):
        return None
    wanted = {normalized(name) for name in names}
    for key, value in block.items():
        if normalized(key) in wanted and isinstance(value, (int, float)):
            return float(value)
    return None


def vector_from_named_block(entry, names):
    if not isinstance(entry, dict):
        return None
    wanted = {normalized(name) for name in names}
    for key, block in entry.items():
        if normalized(key) not in wanted:
            continue
        values = [modality_value(block, modality, index)
                  for index, modality in enumerate(MODALITIES)]
        if all(isinstance(value, (int, float, bool)) for value in values):
            return np.asarray(values, dtype=float)
    return None


def vector_from_modality_subblocks(entry, field_names):
    if not isinstance(entry, dict):
        return None
    values = []
    for index, modality in enumerate(MODALITIES):
        block = modality_value(entry, modality, index)
        value = scalar_from_dict(block, field_names)
        if value is None:
            return None
        values.append(value)
    return np.asarray(values, dtype=float)


def extract_round_history(coverage):
    records = numeric_round_items(coverage["per_round"])
    rounds = []
    contributors = []
    stored_staleness = []

    for fallback_round, entry in records:
        if not isinstance(entry, dict):
            raise ValueError("per_round entry is not an object: {!r}".format(entry))

        round_number = entry.get("round", entry.get("round_index", fallback_round))
        rounds.append(int(round_number))

        contributor_vector = vector_from_named_block(
            entry,
            ("contributor_counts", "contributors", "encoder_contributors"),
        )
        if contributor_vector is None:
            contributor_vector = vector_from_modality_subblocks(
                entry, ("contributors", "contributor_count", "count")
            )
        if contributor_vector is None:
            raise KeyError(
                "Cannot find four encoder contributor counts in round {}. Keys={}".format(
                    round_number, sorted(entry)
                )
            )
        contributors.append(contributor_vector)

        stale_vector = vector_from_named_block(
            entry, ("staleness", "encoder_staleness", "staleness_per_encoder")
        )
        if stale_vector is None:
            stale_vector = vector_from_modality_subblocks(entry, ("staleness", "stale"))
        stored_staleness.append(stale_vector)

    contributor_array = np.vstack(contributors)

    derived = np.zeros_like(contributor_array, dtype=float)
    running = np.zeros(len(MODALITIES), dtype=float)
    for row_index, row in enumerate(contributor_array):
        running = np.where(row > 0, 0.0, running + 1.0)
        derived[row_index] = running

    if stored_staleness and all(value is not None for value in stored_staleness):
        stored = np.vstack(stored_staleness)
        if not np.array_equal(stored, derived):
            print(
                "WARNING: stored encoder staleness differs from the contributor-derived "
                "after-round convention; plotting the stored values"
            )
        staleness = stored
    else:
        staleness = derived

    return np.asarray(rounds), contributor_array, staleness, records


def numeric_client_vector(value):
    if isinstance(value, list) and len(value) == 8:
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return np.asarray(value, dtype=float)
    if isinstance(value, dict) and len(value) == 8:
        def client_number(key):
            digits = re.findall(r"\d+", str(key))
            return int(digits[-1]) if digits else 999

        keys = sorted(value, key=client_number)
        vals = [value[key] for key in keys]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in vals):
            return np.asarray(vals, dtype=float)
    return None


def active_clients_from_records(records):
    counts = np.zeros(8, dtype=float)
    saw_active = False
    all_ids = []
    rows = []
    for _, entry in records:
        active = None
        if isinstance(entry, dict):
            for key in ("active_clients", "selected_clients", "participants"):
                if key in entry and isinstance(entry[key], list):
                    active = entry[key]
                    break
        if active is None:
            continue
        active = [int(value) for value in active]
        rows.append(active)
        all_ids.extend(active)
        saw_active = True
    if not saw_active:
        return None
    zero_based = 0 in all_ids
    for active in rows:
        for client in active:
            index = client if zero_based else client - 1
            if not 0 <= index < 8:
                raise ValueError("Invalid active client id {}".format(client))
            counts[index] += 1
    return counts


def rounds_participated(metrics, records):
    candidates = []
    for key in ("rounds_participated", "participation_counts", "client_participation_counts"):
        candidates.extend(find_key(metrics, key))
    vectors = []
    for path, value in candidates:
        vector = numeric_client_vector(value)
        if vector is not None:
            vectors.append((path, vector))
    if vectors:
        return max(vectors, key=lambda item: float(np.sum(item[1])))
    derived = active_clients_from_records(records)
    if derived is None:
        raise KeyError("rounds_participated and per-round active_clients are both absent")
    return "derived from encoder_coverage.per_round.active_clients", derived


def jain_index(values):
    values = np.asarray(values, dtype=float)
    denominator = len(values) * np.sum(values ** 2)
    return float(np.sum(values) ** 2 / denominator) if denominator else float("nan")


def gini_coefficient(values):
    values = np.asarray(values, dtype=float)
    denominator = 2.0 * len(values) * np.sum(values)
    if denominator == 0:
        return float("nan")
    return float(np.abs(values[:, None] - values[None, :]).sum() / denominator)


def predicted_idle_probability(n_clients, pool_size, k):
    unavailable = n_clients - pool_size
    if unavailable < k:
        return 0.0
    return math.comb(unavailable, k) / float(math.comb(n_clients, k))


def final_round_report(metrics):
    rounds = metrics.get("rounds")
    if not isinstance(rounds, dict):
        return metrics
    keys = [key for key in rounds if str(key).isdigit()]
    if not keys:
        return metrics
    return rounds[max(keys, key=lambda key: int(key))]


def find_rounds_to_target(metrics):
    report = final_round_report(metrics)
    matches = []
    for key, value in report.items() if isinstance(report, dict) else []:
        if "rounds_to_target_dice" in key:
            matches.append(("$.final.{}".format(key), value))
    if matches:
        return matches
    return [item for item in find_key(metrics, "rounds_to_target_dice")]


def flatten_round_values(value, prefix=""):
    rows = []
    if value is None:
        rows.append((prefix or "value", None))
    elif isinstance(value, bool):
        return rows
    elif isinstance(value, (int, float)):
        rows.append((prefix or "value", float(value)))
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            rows.extend(flatten_round_values(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = "{}[{}]".format(prefix, index) if prefix else "[{}]".format(index)
            rows.extend(flatten_round_values(child, child_prefix))
    return rows


def save_figure(fig, out_dir, stem):
    if VALUES_ONLY:
        plt.close(fig)
        return
    fig.savefig(out_dir / (stem + ".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    global VALUES_ONLY
    args = parse_args()
    VALUES_ONLY = args.values_only
    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for split, seed, task in RUN_GRID:
        result = locate_result(root, split, seed, task, args.array_job, args.k)
        with (result / "metrics.json").open() as handle:
            metrics = json.load(handle)
        coverage_path, coverage = choose_coverage(metrics)
        rounds, contributors, staleness, records = extract_round_history(coverage)
        participation_path, participation = rounds_participated(metrics, records)

        if len(rounds) != args.rounds:
            raise AssertionError(
                "{} has {} coverage rounds, expected {}".format(
                    result, len(rounds), args.rounds
                )
            )
        if int(np.sum(participation)) != args.k * args.rounds:
            raise AssertionError(
                "{} participation sums to {}, expected {}".format(
                    result, int(np.sum(participation)), args.k * args.rounds
                )
            )

        runs[(split, seed)] = {
            "result": result,
            "metrics": metrics,
            "coverage_path": coverage_path,
            "rounds": rounds,
            "contributors": contributors,
            "staleness": staleness,
            "participation_path": participation_path,
            "participation": participation,
        }

    # 1. Staleness: representative Scenario 1 seed 42, plus a complete CSV.
    with (out_dir / "staleness_per_round_all_splitA_seeds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "seed", "round", "encoder", "contributors", "staleness"])
        for seed in (42, 123, 2026):
            run = runs[("A", seed)]
            for row, round_number in enumerate(run["rounds"]):
                for encoder, modality in enumerate(MODALITIES):
                    writer.writerow([
                        "A", seed, int(round_number), modality,
                        int(run["contributors"][row, encoder]),
                        int(run["staleness"][row, encoder]),
                    ])

    representative = runs[("A", 42)]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = ("#4C78A8", "#E45756", "#54A24B", "#F2CF5B")
    for index, (modality, color) in enumerate(zip(MODALITIES, colors)):
        ax.step(
            representative["rounds"] + 1,
            representative["staleness"][:, index],
            where="post",
            label=modality,
            color=color,
            linewidth=1.6,
        )
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Encoder staleness (rounds)")
    ax.set_title("Scenario 1, K=2, seed 42")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    save_figure(fig, out_dir, "encoder_staleness_splitA_K2_seed42")

    with (out_dir / "encoder_staleness_maxima.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "seed", "encoder", "max_staleness", "idle_rounds"])
        for (split, seed), run in sorted(runs.items()):
            for index, modality in enumerate(MODALITIES):
                writer.writerow([
                    split, seed, modality,
                    int(np.max(run["staleness"][:, index])),
                    int(np.sum(run["contributors"][:, index] == 0)),
                ])

    # 2. Participation fairness over all six runs.
    fairness_rows = []
    for (split, seed), run in sorted(runs.items()):
        values = run["participation"]
        fairness_rows.append({
            "split": split,
            "seed": seed,
            "jain_index": jain_index(values),
            "gini": gini_coefficient(values),
            "minimum": int(np.min(values)),
            "maximum": int(np.max(values)),
            "range": int(np.max(values) - np.min(values)),
            "mean": float(np.mean(values)),
            "rounds_participated": [int(value) for value in values],
            "source": run["participation_path"],
        })

    with (out_dir / "participation_fairness.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "split", "seed", "jain_index", "gini", "minimum", "maximum",
            "range", "mean", "rounds_participated", "source",
        ])
        for row in fairness_rows:
            writer.writerow([
                row["split"], row["seed"], row["jain_index"], row["gini"],
                row["minimum"], row["maximum"], row["range"], row["mean"],
                json.dumps(row["rounds_participated"]), row["source"],
            ])

    labels = ["{}-{}".format(row["split"], row["seed"]) for row in fairness_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].bar(x, [row["jain_index"] for row in fairness_rows], color="#4C78A8")
    axes[0].set_ylabel("Jain's fairness index")
    axes[0].set_ylim(0.8, 1.005)
    axes[1].bar(x, [row["gini"] for row in fairness_rows], color="#E45756")
    axes[1].set_ylabel("Gini coefficient")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Client-participation fairness, K=2")
    fig.tight_layout()
    save_figure(fig, out_dir, "participation_fairness_K2_all_runs")

    # 3. Bytes to target Dice. Preserve the full JSON path so no target/entity
    # semantics are guessed by this post-processing script.
    bytes_rows = []
    bytes_per_round = args.mb_per_round * 1_000_000.0
    for (split, seed), run in sorted(runs.items()):
        matches = find_rounds_to_target(run["metrics"])
        for base_path, block in matches:
            for leaf_path, rounds_value in flatten_round_values(block):
                byte_value = None if rounds_value is None else rounds_value * bytes_per_round
                bytes_rows.append({
                    "split": split,
                    "seed": seed,
                    "metric_path": "{}.{}".format(base_path, leaf_path),
                    "rounds_to_target": rounds_value,
                    "bytes_to_target": byte_value,
                    "decimal_GB_to_target": None if byte_value is None else byte_value / 1e9,
                    "GiB_to_target": None if byte_value is None else byte_value / (1024.0 ** 3),
                })

    with (out_dir / "bytes_to_target_dice.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "seed", "metric_path", "rounds_to_target",
            "bytes_to_target", "decimal_GB_to_target", "GiB_to_target",
        ))
        writer.writeheader()
        writer.writerows(bytes_rows)

    finite_bytes_rows = [
        row for row in bytes_rows
        if row["rounds_to_target"] is not None
        and 0 <= row["rounds_to_target"] <= args.rounds
    ]
    if finite_bytes_rows:
        metric_paths = sorted({row["metric_path"] for row in finite_bytes_rows})
        x = np.arange(len(metric_paths))
        fig, axes = plt.subplots(1, 2, figsize=(max(9.0, len(x) * 0.8), 4.2), sharey=True)
        for ax, split in zip(axes, ("A", "B")):
            for seed, marker in zip((42, 123, 2026), ("o", "s", "^")):
                lookup = {
                    row["metric_path"]: row["decimal_GB_to_target"]
                    for row in finite_bytes_rows
                    if row["split"] == split and row["seed"] == seed
                }
                y = [lookup.get(path, np.nan) for path in metric_paths]
                ax.plot(x, y, marker=marker, linewidth=1.3, label="Seed {}".format(seed))
            ax.set_xticks(x)
            ax.set_xticklabels(
                [path.split("rounds_to_target_dice", 1)[-1].lstrip("._") or path
                 for path in metric_paths],
                rotation=45,
                ha="right",
            )
            ax.set_title("Scenario {} / Split {}".format(1 if split == "A" else 2, split))
            ax.set_xlabel("Target-Dice criterion")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Communication to target (decimal GB)")
        axes[0].legend(frameon=False)
        fig.suptitle("Communication efficiency at {:.0f} MB per round".format(args.mb_per_round))
        fig.tight_layout()
        save_figure(fig, out_dir, "bytes_to_target_dice_K2")

    # 4. Coverage: predicted versus observed, seed SD as the error bar.
    coverage_rows = []
    for split in ("A", "B"):
        predicted_idle = np.asarray([
            predicted_idle_probability(8, pool, args.k) for pool in POOL_SIZES[split]
        ])
        observed_coverage_by_seed = np.vstack([
            np.mean(runs[(split, seed)]["contributors"] > 0, axis=0)
            for seed in (42, 123, 2026)
        ])
        observed_mean = np.mean(observed_coverage_by_seed, axis=0)
        observed_sd = np.std(observed_coverage_by_seed, axis=0, ddof=1)
        predicted_coverage = 1.0 - predicted_idle

        for index, modality in enumerate(MODALITIES):
            coverage_rows.append({
                "split": split,
                "encoder": modality,
                "pool_size": POOL_SIZES[split][index],
                "predicted_idle_fraction": predicted_idle[index],
                "predicted_coverage_fraction": predicted_coverage[index],
                "observed_coverage_seed42": observed_coverage_by_seed[0, index],
                "observed_coverage_seed123": observed_coverage_by_seed[1, index],
                "observed_coverage_seed2026": observed_coverage_by_seed[2, index],
                "observed_coverage_mean": observed_mean[index],
                "observed_coverage_sample_sd": observed_sd[index],
                "observed_minus_predicted": observed_mean[index] - predicted_coverage[index],
            })

    with (out_dir / "encoder_coverage_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    x = np.arange(len(MODALITIES))
    for ax, split in zip(axes, ("A", "B")):
        rows = [row for row in coverage_rows if row["split"] == split]
        predicted = np.asarray([row["predicted_coverage_fraction"] for row in rows]) * 100
        observed = np.asarray([row["observed_coverage_mean"] for row in rows]) * 100
        sd = np.asarray([row["observed_coverage_sample_sd"] for row in rows]) * 100
        ax.plot(x, predicted, "s--", color="#222222", label="Predicted", zorder=3)
        ax.errorbar(
            x, observed, yerr=np.vstack([sd, sd]), fmt="o", color="#4C78A8",
            ecolor="#4C78A8", capsize=5, linewidth=1.5,
            label="Observed mean ± SD", zorder=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(MODALITIES)
        ax.set_title("Scenario {} / Split {}".format(1 if split == "A" else 2, split))
        ax.set_xlabel("Modality-specific encoder")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Rounds with an encoder update (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("Predicted versus observed encoder coverage, K=2, n=3 seeds")
    fig.tight_layout()
    save_figure(fig, out_dir, "encoder_coverage_validation_K2")

    print("\nSELECTION-MECHANISM ANALYSIS PASSED")
    print("Output directory:", out_dir)
    print("\nStaleness maxima:")
    for index, modality in enumerate(MODALITIES):
        print("  {:5s}: max={}, idle_rounds={}".format(
            modality,
            int(np.max(representative["staleness"][:, index])),
            int(np.sum(representative["contributors"][:, index] == 0)),
        ))
    print("\nParticipation fairness:")
    for row in fairness_rows:
        print(
            "  Split {} seed {:4d}: Jain={:.6f}, Gini={:.6f}, range={}..{}, counts={}".format(
                row["split"], row["seed"], row["jain_index"], row["gini"],
                row["minimum"], row["maximum"], row["rounds_participated"],
            )
        )
    print("\nBytes-to-target rows:", len(bytes_rows))
    print("Communication assumption: {:.3f} MB/round (decimal MB)".format(args.mb_per_round))


if __name__ == "__main__":
    main()
