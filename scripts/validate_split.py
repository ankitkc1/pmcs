#!/usr/bin/env python3
"""Validate paired, grade-stratified federated client split configs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set


MODALITY_ORDER = ["FLAIR", "T1ce", "T1", "T2"]
REPORT_ORDER = ["FLAIR", "T1", "T2", "T1ce"]
PARTITIONS = ["train", "validation", "test"]
GRADES = ["HGG", "LGG"]
EXPECTED_TARGETS = {
    "A": {"FLAIR": 7, "T1ce": 2, "T1": 5, "T2": 4},
    "B": {"FLAIR": 2, "T1ce": 7, "T1": 5, "T2": 4},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    require(config.get("schema_version") == 1, f"{path}: unsupported schema")
    return config


def expand_specs(grade: str, specs: Sequence[str]) -> List[str]:
    result: List[str] = []
    for spec in specs:
        parts = spec.split("-")
        require(len(parts) in (1, 2), f"invalid range {grade}:{spec}")
        start = int(parts[0])
        end = int(parts[-1])
        require(start <= end, f"reversed range {grade}:{spec}")
        result.extend(f"{grade}_{index:03d}" for index in range(start, end + 1))
    require(len(result) == len(set(result)), f"duplicate slot in {grade}:{specs}")
    return result


def section_slots(section: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    return {grade: expand_specs(grade, section.get(grade, [])) for grade in GRADES}


def flattened(slots: Mapping[str, Sequence[str]]) -> Set[str]:
    return {slot for grade in GRADES for slot in slots[grade]}


def idle_probability(num_clients: int, providers: int, selected: int) -> float:
    unavailable = num_clients - providers
    if unavailable < selected:
        return 0.0
    return math.comb(unavailable, selected) / math.comb(num_clients, selected)


def assignment_signature(config: Mapping) -> dict:
    assignment = config["assignment"]
    clients = {}
    for client_id, client in assignment["clients"].items():
        clients[client_id] = {
            partition: {
                grade: sorted(section_slots(client[partition])[grade])
                for grade in GRADES
            }
            for partition in PARTITIONS
        }
    return {
        "data_seed": assignment["data_seed"],
        "algorithm": assignment["algorithm"],
        "global_test": {
            grade: sorted(
                expand_specs(grade, config["global_test"]["grade_slot_ranges"][grade])
            )
            for grade in GRADES
        },
        "clients": clients,
    }


def patient_rank(data_seed: int, grade: str, patient_id: str) -> bytes:
    """Return a stable seeded rank that is independent of model/client RNG state."""
    payload = "federated-split-v1:{}:{}:{}".format(
        data_seed, grade, patient_id
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def volume_case_id(path: Path) -> str:
    suffix = "_vol.npy"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def inspect_volumes(config: dict, volume_dir: Path) -> Dict[str, bool]:
    """Return case -> full-modal flag after checking every preprocessed array."""
    try:
        import numpy as np
    except ImportError as exc:
        raise AssertionError("--volume_dir validation requires numpy") from exc

    expected_shape = tuple(config["assignment"]["expected_volume_shape"])
    result: Dict[str, bool] = {}
    for path in sorted(volume_dir.glob("*.npy")):
        patient_id = volume_case_id(path)
        require(patient_id not in result, "duplicate volume for {}".format(patient_id))
        shape = tuple(np.load(str(path), mmap_mode="r").shape)
        result[patient_id] = shape == expected_shape
    require(result, "{}: no .npy volumes found".format(volume_dir))
    require(
        all(result.values()),
        "{}: one or more volumes do not have expected shape {}".format(
            volume_dir, expected_shape
        ),
    )
    return result


def materialize_patient_slots(
    config: dict,
    manifest_path: Path,
    volume_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    """Map symbolic grade slots to real patients using only assignment.data_seed."""
    assignment = config["assignment"]
    columns = assignment["manifest_columns"]
    required = {columns["patient_id"], columns["grade"]}
    modality_column = columns.get("all_modalities_present")
    if volume_dir is None:
        require(
            modality_column is not None,
            "manifest has no modality-presence column; provide --volume_dir",
        )
        required.add(modality_column)
    volume_index = inspect_volumes(config, volume_dir) if volume_dir is not None else None
    grouped = {grade: [] for grade in GRADES}
    seen_ids: Set[str] = set()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "{}: missing CSV header".format(manifest_path))
        require(required <= set(reader.fieldnames), "{}: missing required columns {}".format(
            manifest_path, sorted(required - set(reader.fieldnames))
        ))
        for line_number, row in enumerate(reader, start=2):
            patient_id = row[columns["patient_id"]].strip()
            grade = row[columns["grade"]].strip().upper()
            require(patient_id != "", "{}:{}: empty patient_id".format(manifest_path, line_number))
            if volume_index is not None and patient_id not in volume_index:
                continue
            require(patient_id not in seen_ids, "{}: duplicate patient {}".format(manifest_path, patient_id))
            require(grade in GRADES, "{}:{}: grade must be HGG or LGG".format(manifest_path, line_number))
            if modality_column is not None:
                present_text = row[modality_column].strip().lower()
                require(
                    present_text in {"1", "0", "true", "false", "yes", "no"},
                    "{}:{}: invalid all_modalities_present value".format(
                        manifest_path, line_number
                    ),
                )
                all_modalities_present = present_text in {"1", "true", "yes"}
            else:
                all_modalities_present = bool(volume_index[patient_id])
            seen_ids.add(patient_id)
            grouped[grade].append({
                "patient_id": patient_id,
                "all_modalities_present": all_modalities_present,
            })

    if volume_index is not None:
        require(
            seen_ids == set(volume_index),
            "mapping/volume mismatch: volumes without mapped grade={}".format(
                sorted(set(volume_index) - seen_ids)
            ),
        )

    expected_counts = assignment["population_grade_counts"]
    actual_counts = {grade: len(grouped[grade]) for grade in GRADES}
    require(actual_counts == expected_counts, "{}: grade counts {} != {}".format(
        manifest_path, actual_counts, expected_counts
    ))

    slot_map: Dict[str, dict] = {}
    for grade in GRADES:
        patients = sorted(grouped[grade], key=lambda item: item["patient_id"])
        patients.sort(
            key=lambda item: (
                patient_rank(assignment["data_seed"], grade, item["patient_id"]),
                item["patient_id"],
            )
        )
        for index, patient in enumerate(patients):
            slot_map["{}_{:03d}".format(grade, index)] = patient
    return slot_map


def mapping_fingerprint(slot_map: Mapping[str, dict]) -> str:
    assignments = [
        "{}={}".format(slot, slot_map[slot]["patient_id"])
        for slot in sorted(slot_map)
    ]
    return hashlib.sha256("\n".join(assignments).encode("utf-8")).hexdigest()[:16]


def validate_config(
    config: dict,
    path: Path,
    requested_data_seed: Optional[int],
    manifest_path: Optional[Path],
    volume_dir: Optional[Path],
) -> None:
    split_id = config["split_id"]
    require(split_id in EXPECTED_TARGETS, f"{path}: split_id must be A or B")
    require(config["modality_order"] == MODALITY_ORDER, f"{path}: modality order mismatch")
    require(config["num_clients"] == 8, f"{path}: expected 8 clients")

    assignment = config["assignment"]
    require(assignment.get("model_seed_independent") is True, "assignment must ignore --seed")
    require("seed" not in assignment, "model/client-selection seed leaked into assignment")
    if requested_data_seed is not None:
        require(
            requested_data_seed == assignment["data_seed"],
            f"{path}: --data_seed does not match the materialized config",
        )

    clients = config["clients"]
    require(len(clients) == config["num_clients"], f"{path}: client count mismatch")
    client_ids = [client["id"] for client in clients]
    require(len(client_ids) == len(set(client_ids)), f"{path}: duplicate client IDs")
    require(set(client_ids) == set(assignment["clients"]), f"{path}: assignment/client mismatch")

    pool_sizes = dict.fromkeys(MODALITY_ORDER, 0)
    for client in clients:
        mask = client["mask"]
        require(len(mask) == len(MODALITY_ORDER), f"{client['id']}: mask length mismatch")
        require(all(bit in (0, 1) for bit in mask), f"{client['id']}: non-binary mask")
        require(sum(mask) > 0, f"{client['id']}: zero modalities")
        for modality, bit in zip(MODALITY_ORDER, mask):
            pool_sizes[modality] += bit

    target = EXPECTED_TARGETS[split_id]
    require(config["expected_modality_pool_sizes"] == target, f"{path}: stored target mismatch")
    require(pool_sizes == target, f"{path}: calculated pool sizes {pool_sizes} != {target}")

    print(f"\nSplit {split_id}: {path}")
    print(f"data_seed={assignment['data_seed']} (independent of model/client-selection --seed)")
    print("Pool sizes: " + ", ".join(f"{m}={pool_sizes[m]}" for m in MODALITY_ORDER))
    print("Idle probability:")
    print("K | " + " | ".join(f"{m:>6}" for m in REPORT_ORDER))
    for selected in (8, 4, 2):
        values = [100.0 * idle_probability(config["num_clients"], pool_sizes[m], selected) for m in REPORT_ORDER]
        print(f"{selected} | " + " | ".join(f"{value:5.1f}%" for value in values))

    global_test = config["global_test"]
    require(global_test["size"] == 50, f"{path}: global test size must be 50")
    require(global_test["mask"] == [1, 1, 1, 1], f"{path}: global test must be full-modal")
    require(global_test.get("all_modalities_present") is True, f"{path}: full modalities not confirmed")
    global_by_grade = {
        grade: expand_specs(grade, global_test["grade_slot_ranges"][grade]) for grade in GRADES
    }
    require(
        {grade: len(global_by_grade[grade]) for grade in GRADES} == global_test["grade_counts"],
        f"{path}: global grade counts mismatch",
    )
    global_slots = flattened(global_by_grade)
    require(len(global_slots) == 50, f"{path}: expanded global test size mismatch")

    if manifest_path is not None:
        slot_map = materialize_patient_slots(config, manifest_path, volume_dir)
        require(
            all(slot_map[slot]["all_modalities_present"] for slot in global_slots),
            "{}: a selected global-test patient lacks one or more modalities".format(path),
        )
        print(
            "Manifest assignment: {} real patients; fingerprint={}".format(
                len(slot_map), mapping_fingerprint(slot_map)
            )
        )
    else:
        print("Manifest assignment: symbolic slots checked (use --manifest for real patient IDs)")

    print("Client counts (train/validation/test; HGG/LGG):")
    all_client_slots: Set[str] = set()
    all_client_by_grade = {grade: set() for grade in GRADES}
    totals: List[int] = []
    hgg_proportions: List[float] = []
    for client_id in client_ids:
        record = assignment["clients"][client_id]
        partition_sets: Dict[str, Set[str]] = {}
        grade_total = dict.fromkeys(GRADES, 0)
        partition_sizes = []
        for partition in PARTITIONS:
            by_grade = section_slots(record[partition])
            for grade in GRADES:
                grade_total[grade] += len(by_grade[grade])
                all_client_by_grade[grade].update(by_grade[grade])
            partition_sets[partition] = flattened(by_grade)
            partition_sizes.append(len(partition_sets[partition]))

        require(grade_total == record["grade_counts"], f"{client_id}: grade counts mismatch")
        require(all(size > 0 for size in partition_sizes), f"{client_id}: empty local partition")
        require(
            not (partition_sets["train"] & partition_sets["validation"])
            and not (partition_sets["train"] & partition_sets["test"])
            and not (partition_sets["validation"] & partition_sets["test"]),
            f"{client_id}: train/validation/test overlap",
        )
        client_slots = set().union(*partition_sets.values())
        require(not client_slots & global_slots, f"{client_id}: overlaps global test")
        require(not client_slots & all_client_slots, f"{client_id}: overlaps another client")
        all_client_slots.update(client_slots)
        total = len(client_slots)
        totals.append(total)
        hgg_proportions.append(grade_total["HGG"] / total)
        print(
            f"  {client_id}: total={total:2d}, split={partition_sizes[0]}/{partition_sizes[1]}/"
            f"{partition_sizes[2]}, HGG/LGG={grade_total['HGG']}/{grade_total['LGG']}"
        )

    require(max(totals) - min(totals) <= 1, f"{path}: client totals are not maximally equalized")
    population = assignment["population_grade_counts"]
    for grade in GRADES:
        expected_grade_slots = {
            "{}_{:03d}".format(grade, index) for index in range(population[grade])
        }
        assigned_grade_slots = set(global_by_grade[grade]) | all_client_by_grade[grade]
        require(
            assigned_grade_slots == expected_grade_slots,
            f"{path}: {grade} population is not assigned exactly once",
        )
    require(
        max(hgg_proportions) - min(hgg_proportions) <= 0.025000001,
        f"{path}: tumour-grade proportions are not sufficiently balanced",
    )
    expected_total = sum(population.values())
    require(len(global_slots | all_client_slots) == expected_total, f"{path}: population not fully assigned")
    print(
        "Client HGG proportion range: {:.1f}% to {:.1f}%".format(
            100.0 * min(hgg_proportions), 100.0 * max(hgg_proportions)
        )
    )
    print("Zero-modality clients: none")
    print("Global held-out test overlap with clients: none")
    print("Global held-out test: 50 cases, all four modalities present")


def validate_pair(first: dict, second: dict) -> None:
    by_id = {first["split_id"]: first, second["split_id"]: second}
    require(set(by_id) == {"A", "B"}, "paired validation requires one Split A and one Split B")
    split_a, split_b = by_id["A"], by_id["B"]

    require(assignment_signature(split_a) == assignment_signature(split_b), "patient assignment differs")

    masks_a = {client["id"]: client["mask"] for client in split_a["clients"]}
    masks_b = {client["id"]: client["mask"] for client in split_b["clients"]}
    for client_id, mask_a in masks_a.items():
        expected_b = [mask_a[1], mask_a[0], mask_a[2], mask_a[3]]
        require(masks_b[client_id] == expected_b, f"{client_id}: mask is not FLAIR/T1ce swap")

    # Prove that no non-derived experimental setting differs.
    normalized_a = copy.deepcopy(split_a)
    normalized_b = copy.deepcopy(split_b)
    normalized_b["split_id"] = normalized_a["split_id"]
    normalized_b["expected_modality_pool_sizes"] = normalized_a["expected_modality_pool_sizes"]
    for client_a, client_b in zip(normalized_a["clients"], normalized_b["clients"]):
        client_b["mask"] = client_a["mask"]
    require(normalized_a == normalized_b, "configs differ beyond masks and derived pool targets")

    print("\nPaired A/B validation:")
    print("Patient assignment: IDENTICAL")
    print("Global held-out test: IDENTICAL")
    print("Masks: differ only by FLAIR/T1ce bit swap")
    print("All non-derived experimental settings: IDENTICAL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="one config, or Split A and B together")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="assert the fixed data-assignment seed; model --seed is deliberately unsupported",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "optional BraTS name_mapping.csv (or another configured grade manifest)"
        ),
    )
    parser.add_argument(
        "--volume_dir",
        type=Path,
        default=None,
        help="directory containing *_vol.npy arrays; verifies the mapped cases and four modalities",
    )
    args = parser.parse_args()
    require(len(args.configs) in (1, 2), "provide one config or the paired A/B configs")
    return args


def main() -> None:
    args = parse_args()
    loaded = [(path, load_config(path)) for path in args.configs]
    for path, config in loaded:
        validate_config(config, path, args.data_seed, args.manifest, args.volume_dir)
    if len(loaded) == 2:
        validate_pair(loaded[0][1], loaded[1][1])
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
