#!/usr/bin/env python3
"""Materialize paired Split A/B CSVs from real BraTS patient IDs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping

from validate_split import (
    GRADES,
    PARTITIONS,
    assignment_signature,
    load_config,
    mapping_fingerprint,
    materialize_patient_slots,
    require,
    section_slots,
    validate_config,
    validate_pair,
)


def write_lines(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(value + "\n")
    os.replace(str(temp_path), str(path))


def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temp_path), str(path))


def resolve_section(section: Mapping, slot_map: Mapping[str, dict]) -> Dict[str, List[str]]:
    slots = section_slots(section)
    return {
        grade: [slot_map[slot]["patient_id"] for slot in slots[grade]]
        for grade in GRADES
    }


def flatten_by_grade(section: Mapping[str, List[str]]) -> List[str]:
    return [patient_id for grade in GRADES for patient_id in section[grade]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_a", type=Path)
    parser.add_argument("config_b", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--volume_dir", type=Path, required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_a = load_config(args.config_a)
    config_b = load_config(args.config_b)

    validate_config(
        config_a, args.config_a, args.data_seed, args.manifest, args.volume_dir
    )
    validate_config(
        config_b, args.config_b, args.data_seed, args.manifest, args.volume_dir
    )
    validate_pair(config_a, config_b)
    require(
        assignment_signature(config_a) == assignment_signature(config_b),
        "Split A/B symbolic assignments differ",
    )

    slot_map = materialize_patient_slots(config_a, args.manifest, args.volume_dir)
    fingerprint = mapping_fingerprint(slot_map)
    output_root = args.output_dir / "data_seed_{}".format(args.data_seed)

    global_ranges = config_a["global_test"]["grade_slot_ranges"]
    global_slots = section_slots(global_ranges)
    global_patients = {
        grade: [slot_map[slot]["patient_id"] for slot in global_slots[grade]]
        for grade in GRADES
    }
    global_ids = flatten_by_grade(global_patients)
    require(len(global_ids) == 50, "global test must contain exactly 50 patients")
    require(len(global_ids) == len(set(global_ids)), "duplicate global-test patient")
    require(
        all(
            slot_map[slot]["all_modalities_present"]
            for grade in GRADES
            for slot in global_slots[grade]
        ),
        "global test contains a non-full-modal patient",
    )
    global_path = output_root / "global_test.csv"
    write_lines(global_path, global_ids)

    assignment_record = {
        "schema_version": 1,
        "data_seed": args.data_seed,
        "algorithm": config_a["assignment"]["algorithm"],
        "mapping_fingerprint": fingerprint,
        "population_grade_counts": config_a["assignment"]["population_grade_counts"],
        "manifest": str(args.manifest.resolve()),
        "volume_dir": str(args.volume_dir.resolve()),
        "global_test": global_patients,
        "clients": {},
    }

    materialized_paths: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for config in (config_a, config_b):
        split_name = "split{}".format(config["split_id"])
        split_root = output_root / split_name
        materialized_paths[split_name] = {}
        masks = {client["id"]: client["mask"] for client in config["clients"]}
        materialized_config = {
            "schema_version": 1,
            "split_id": config["split_id"],
            "modality_order": config["modality_order"],
            "masks": masks,
            "expected_modality_pool_sizes": config["expected_modality_pool_sizes"],
            "data_seed": args.data_seed,
            "mapping_fingerprint": fingerprint,
            "global_test_file": str(global_path.resolve()),
            "global_test_grade_counts": config["global_test"]["grade_counts"],
            "clients": {},
        }

        for client_id, record in config["assignment"]["clients"].items():
            materialized_paths[split_name][client_id] = {}
            client_record = {
                "grade_counts": record["grade_counts"],
                "files": {},
                "partition_counts": {},
            }
            assignment_record["clients"].setdefault(client_id, {
                "grade_counts": record["grade_counts"],
                "partitions": {},
            })

            for partition in PARTITIONS:
                resolved = resolve_section(record[partition], slot_map)
                patient_ids = flatten_by_grade(resolved)
                csv_path = split_root / "{}_{}.csv".format(client_id, partition)
                write_lines(csv_path, patient_ids)
                materialized_paths[split_name][client_id][partition] = csv_path
                client_record["files"][partition] = str(csv_path.resolve())
                client_record["partition_counts"][partition] = len(patient_ids)

                previous = assignment_record["clients"][client_id]["partitions"].get(partition)
                if previous is None:
                    assignment_record["clients"][client_id]["partitions"][partition] = resolved
                else:
                    require(previous == resolved, "Split A/B patient assignment differs")

            materialized_config["clients"][client_id] = client_record

        write_json(split_root / "materialized_config.json", materialized_config)

    # Byte-for-byte equality proves every local partition is identical across A/B.
    for client_id in assignment_record["clients"]:
        for partition in PARTITIONS:
            path_a = materialized_paths["splitA"][client_id][partition]
            path_b = materialized_paths["splitB"][client_id][partition]
            require(path_a.read_bytes() == path_b.read_bytes(), "A/B CSV mismatch")

    write_json(output_root / "assignment.json", assignment_record)

    print("\nMATERIALIZATION PASSED")
    print("Output: {}".format(output_root.resolve()))
    print("Mapping fingerprint: {}".format(fingerprint))
    print("Global test: 50 cases (HGG=40, LGG=10), full-modal and disjoint")
    print("All Split A/B client partition CSVs: byte-for-byte identical")


if __name__ == "__main__":
    main()
