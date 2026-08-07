from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignment_manifold.causal import run_causal
from alignment_manifold.config import load_config, sha256_file
from alignment_manifold.extract import extract_checkpoint
from alignment_manifold.geometry import run_geometry
from alignment_manifold.modeling import resolve_checkpoint
from alignment_manifold.prompts import build_expanded_records, build_smoke_records, write_jsonl
from alignment_manifold.provenance import runtime_manifest, write_json
from alignment_manifold.trajectory import run_trajectory_geometry


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompts(config: dict[str, Any], force: bool) -> None:
    path = Path(config["data"]["path"])
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() and manifest_path.exists() and not force:
        print(f"Using existing prompts: {path}")
        return
    builder = config["data"].get("builder", "smoke")
    if builder == "smoke":
        records = build_smoke_records()
    elif builder == "expanded":
        records = build_expanded_records()
    else:
        raise ValueError(f"Unknown prompt builder: {builder}")
    record_hash = write_jsonl(records, path)
    manifest = {
        "kind": "deterministic_smoke_prompt_set",
        "examples": len(records),
        "clusters": len({record["cluster_id"] for record in records}),
        "categories": {
            category: sum(record["category"] == category for record in records)
            for category in sorted({record["category"] for record in records})
        },
        "record_hash": record_hash,
        "file_sha256": sha256_file(path),
        "seed": config["experiment"]["seed"],
    }
    write_json(manifest, manifest_path)
    print(f"Wrote {len(records)} prompts to {path}")


def _preflight(config: dict[str, Any]) -> None:
    checkpoints = {
        name: {**resolve_checkpoint(value["repo_id"], value["revision"]), "role": value["role"]}
        for name, value in config["checkpoints"].items()
    }
    result = {
        "experiment": config["experiment"]["name"],
        "checkpoints": checkpoints,
        "runtime": runtime_manifest(),
    }
    output = Path(config["experiment"]["artifact_dir"]) / "preflight.json"
    write_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alignment-manifold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Resolve checkpoints and record hardware")
    preflight.add_argument("--config", required=True)

    prompts = subparsers.add_parser("prompts", help="Prompt dataset operations")
    prompt_subparsers = prompts.add_subparsers(dest="prompt_command", required=True)
    prompt_build = prompt_subparsers.add_parser("build")
    prompt_build.add_argument("--config", required=True)
    prompt_build.add_argument("--force", action="store_true")

    extract = subparsers.add_parser("extract", help="Extract one checkpoint sequentially")
    extract.add_argument("--config", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--force", action="store_true")

    geometry = subparsers.add_parser("geometry", help="Fit and compare geometry models")
    geometry.add_argument("--config", required=True)
    geometry.add_argument("--force", action="store_true")

    causal = subparsers.add_parser("causal", help="Run bidirectional single-layer interventions")
    causal.add_argument("--config", required=True)
    causal.add_argument("--force", action="store_true")

    trajectory = subparsers.add_parser(
        "trajectory", help="Fit geometry across an ordered checkpoint trajectory"
    )
    trajectory.add_argument("--config", required=True)
    trajectory.add_argument("--force", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run the full two-checkpoint smoke experiment")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    _seed_everything(int(config["experiment"]["seed"]))
    if args.command == "preflight":
        _preflight(config)
    elif args.command == "prompts":
        _build_prompts(config, args.force)
    elif args.command == "extract":
        extract_checkpoint(config, args.checkpoint, force=args.force)
    elif args.command == "geometry":
        run_geometry(config, force=args.force)
    elif args.command == "causal":
        run_causal(config, force=args.force)
    elif args.command == "trajectory":
        run_trajectory_geometry(config, force=args.force)
    elif args.command == "smoke":
        _build_prompts(config, args.force)
        extract_checkpoint(config, "sft", force=args.force)
        extract_checkpoint(config, "dpo", force=args.force)
        run_geometry(config, force=args.force)
        run_causal(config, force=args.force)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
