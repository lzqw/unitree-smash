#!/usr/bin/env python3
"""Write the remaining native collection plan for branch B1.

The plan is a quota document for the next Isaac capture.  It never turns
control-boundary events into contact labels and never marks an incomplete
archive as a finished pilot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_BUCKETS = {
    "center_5x5": 400,
    "frame_edge": 400,
    "normal_relative_speed": 300,
    "tangential_relative_speed": 250,
    "racket_linear_velocity": 200,
    "racket_angular_velocity": 200,
    "toi_phase": 150,
    "repeat_contact": 100,
}


def make_plan(manifest: dict) -> dict:
    if manifest.get("schema") != "supermimic-racket-ball-contact-pilot-v1":
        raise ValueError("unexpected pilot manifest schema")
    requested_contacts = int(manifest.get("requested_contacts", 2000))
    selected_contacts = int(manifest.get("selected_contacts", 0))
    requested_negatives = int(manifest.get("requested_negatives", 1000))
    selected_negatives = int(manifest.get("selected_negatives", 0))
    coverage = manifest.get("coverage", {})
    split = manifest.get("split", {})
    return {
        "schema": "supermimic-racket-ball-contact-pilot-plan-v1",
        "source_manifest_schema": manifest["schema"],
        "target": {"dynamic_contacts": requested_contacts, "near_miss": requested_negatives},
        "already_measured": {"dynamic_contacts": selected_contacts, "near_miss": selected_negatives},
        "remaining": {
            "dynamic_contacts": max(0, requested_contacts - selected_contacts),
            "near_miss": max(0, requested_negatives - selected_negatives),
        },
        "coverage_quotas": REQUIRED_BUCKETS,
        "coverage_audit": coverage,
        "split_audit": split,
        "required_fields": manifest.get("fields", []),
        "capture_requirements": [
            "native PhysX substeps at 0.0625 ms or the locked task physics dt",
            "TOI pre/post states over at least 8 ms on each side when available",
            "racket part and manifold normal from the native sensor when exposed",
            "contact order and effective mass retained for repeated contacts",
            "ground and net sensor streams retained beside racket contact",
            "negative/no-contact windows sampled independently of the contact rows",
        ],
        "parallelism": {
            "cpu_trace_workers": 8,
            "gpu_capture_workers": 2,
            "gpu_policy": "one bounded Isaac pair per worker, no learner updates",
        },
        "admission": manifest.get("admission", {
            "policy_training_admitted": False,
            "contact_bptt": "detached",
            "v13_residual_gradient": "detached",
        }),
        "status": ("READY_FOR_NATIVE_CAPTURE" if manifest.get("pilot_status") != "READY"
                   else "PILOT_COMPLETE"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
