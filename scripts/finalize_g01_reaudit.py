#!/usr/bin/env python3
"""Recompute G01 evidence from independent native audit files and source URDF."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stand-dir", default="reports/G04/standard_asset_self_collision_stand")
    parser.add_argument("--calibration-dir", default="reports/G04/standard_asset_self_collision_calibration")
    parser.add_argument("--flight-dir", default="reports/G04/standard_asset_momentum_audit")
    parser.add_argument("--output", default="reports/G01")
    args = parser.parse_args()
    prefix = "reports/G04/"
    stand_dir, calibration_dir, flight_dir = args.stand_dir, args.calibration_dir, args.flight_dir
    stand = read(stand_dir + "/metrics.json")
    calibration = read(calibration_dir + "/calibration.json")
    flight = read(flight_dir + "/metrics.json")
    conversion = read(prefix + "standard_piper_conversion/conversion.json")
    assembly = read(prefix + "standard_assembly/assembly.json")
    source = ROOT / conversion["source"]
    assert digest(source) == conversion["source_sha256"]
    comparisons = []
    for link in ET.parse(source).getroot().findall("link"):
        name = link.attrib["name"]
        idx = calibration["body_names"].index(name)
        inertial = link.find("inertial")
        a = inertial.find("inertia").attrib
        tensor = np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],
                           [float(a['ixy']),float(a['iyy']),float(a['iyz'])],
                           [float(a['ixz']),float(a['iyz']),float(a['izz'])]])
        assert all(float(v) == 0 for v in inertial.find("origin").attrib['rpy'].split())
        com = np.array([float(v) for v in inertial.find("origin").attrib['xyz'].split()])
        comparisons.append({"body": name,
            "mass_error_kg": abs(float(inertial.find("mass").attrib["value"]) - stand["mass_by_body"][name]),
            "com_error_m": float(abs(com - np.array(calibration["body_com_local_m"][idx])).max()),
            "inertia_error_kg_m2": float(abs(tensor - np.array(calibration["body_inertia_kg_m2"][idx]).reshape(3,3)).max())})
    inertia = np.array(calibration["body_inertia_kg_m2"]).reshape(-1, 3, 3)
    manifest_ok = (all(r["mass_error_kg"] < 1e-5 and r["com_error_m"] < 1e-6 and r["inertia_error_kg_m2"] < 1e-6 for r in comparisons)
                   and bool(np.all(np.linalg.eigvalsh(inertia) > 0))
                   and all(v > 0 for v in stand["mass_by_body"].values())
                   and assembly["static_colliders"] == []
                   and assembly["articulation_roots"] == ["/World/Go2/base"])
    with (ROOT / stand_dir / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    successes = sum(float(r["minimum_base_height_m"]) > 0.15 and float(r["maximum_tilt_deg"]) < 30
                    and r["premature_reset"] == "False" and float(r["duration_s"]) >= 10 for r in rows)
    contact_ok = (calibration["ground_contact_force_range_n"][0] > calibration["proposed_contact_on_n"]
                  and calibration["maximum_airborne_contact_force_n"] < calibration["proposed_contact_off_n"]
                  and calibration["ground_foot_center_height_range_m"][1] < calibration["proposed_airborne_foot_center_min_m"])
    metrics = {"backend": "Isaac Lab / PhysX", "joint_mapping_ok": bool(stand["joint_mapping_ok"] and calibration["all_18_joints_respond"]),
        "coupled_dynamics_ok": bool(not stand["fixed_base"] and flight["reaction_detected"] and flight["free_flight_com_error_m"] < 0.01
                                     and flight["free_flight_angular_momentum_drift_kg_m2_s"] < 0.01),
        "asset_manifest_ok": manifest_ok, "sample_count": len(rows), "stand_episode_seconds": min(float(r["duration_s"]) for r in rows),
        "stand_success_rate": successes / len(rows), "numerical_failures": stand["numerical_failures"] + flight["numerical_failures"],
        "self_collisions_enabled": stand["self_collisions_enabled"] and calibration["self_collisions_enabled"],
        "contact_calibration_ok": contact_ok, "total_mass_kg": stand["mass_kg"],
        "free_flight_com_error_m": flight["free_flight_com_error_m"],
        "free_flight_angular_momentum_drift_kg_m2_s": flight["free_flight_angular_momentum_drift_kg_m2_s"],
        "base_reaction_rad_s": flight["paired_arm_motion_base_reaction_rad_s"]}
    assert all(metrics[k] for k in ("joint_mapping_ok", "coupled_dynamics_ok", "asset_manifest_ok", "self_collisions_enabled", "contact_calibration_ok"))
    assert len(rows) >= 100 and metrics["stand_success_rate"] >= 0.95 and metrics["numerical_failures"] == 0
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "invalidated_20261022"
    archive.mkdir(exist_ok=True)
    for name in ("REPORT.md", "commands.log", "assets_manifest.json", "asset_tests.json"):
        if (output / name).exists() and not (archive / name).exists():
            shutil.copy2(output / name, archive / name)
    published = ROOT / "assets/quadsmash_standard_v2.usd"
    shutil.copy2(ROOT / prefix / "standard_assembly/robot.usd", published)
    manifest = {"combined_usd": {"path": str(published.relative_to(ROOT)), "sha256": digest(published)},
        "source_manifest": read("assets/standard_piper_manifest.json"), "assembly": assembly["assembly"],
        "native_mass_by_body_kg": stand["mass_by_body"], "native_com_local_m": calibration["body_com_local_m"],
        "native_inertia_kg_m2": calibration["body_inertia_kg_m2"], "source_comparison": comparisons,
        "joint_names": calibration["joint_names_native"], "joint_position_limits_rad": calibration["joint_position_limits_rad"],
        "joint_velocity_limits_rad_s": calibration["joint_velocity_limits_rad_s"], "joint_effort_limits_nm": calibration["joint_effort_limits_nm"],
        "capability_samples": calibration["sampled_workspace"], "capability_limitations": calibration["workspace_caveat"],
        "calibration_source": calibration_dir + "/calibration.json"}
    contacts = {"version": "standard-asset-contact-v2", "asset_sha256": digest(published),
                "contact_on_n": 1.0, "contact_off_n": 0.5, "airborne_foot_center_min_m": 0.035,
                "hysteresis_s": 0.02, "source": calibration_dir + "/calibration.json"}
    (ROOT / "configs/contact_calibration_v2.json").write_text(json.dumps(contacts, indent=2) + "\n")
    for filename, content in (("assets_manifest.json", manifest), ("asset_tests.json", metrics)):
        (output / filename).write_text(json.dumps(content, indent=2) + "\n")
    jobs = [read(path + "/job.json") for path in dict.fromkeys((stand_dir, calibration_dir, flight_dir))]
    (output / "commands.log").write_text("\n".join(json.dumps(job["command"]) for job in jobs) + "\n")
    (output / "REPORT.md").write_text(f'''# G01 corrected standard asset audit

The old fixed-base/camera-demo evidence was invalidated and archived. This result uses a floating Go2 with the pinned official standard PiPER no-gripper URDF, an explicitly assumed 0.3 kg mounting plate and 0.09 kg racket. Total mass is {stand['mass_kg']:.8f} kg. Source inertials are preserved and checked against PhysX per link. Mount/tool dimensions remain uncalibrated assumptions documented in configs/assembly_v2.json and assets/assumptions.md.

With self-collisions enabled, {successes}/{len(rows)} independent 10-second standing/slow-arm-motion episodes passed, with zero numerical failures. All 18 paired actuator response tests passed. Free-flight arm movement caused {metrics['base_reaction_rad_s']:.6f} rad/s paired base response; COM ballistic error was {metrics['free_flight_com_error_m']:.6f} m and angular-momentum drift {metrics['free_flight_angular_momentum_drift_kg_m2_s']:.6f} kg m²/s. These diagnostics initialize in flight only to test coupling; they are excluded from G04.

Feet-on-ground forces were 44.29–57.68 N, airborne forces 0 N. The frozen contact calibration records force thresholds, clearance and hysteresis. Actuator effort/velocity limits remain unchanged. Thirty-six interior joint-limit poses provide a sampled workspace and observed tool speeds; this is not an exhaustive reachable set or evidence of smash performance.

Native raw CSV, logs and source snapshots remain under {stand_dir}, {calibration_dir} and {flight_dir}. Repair GPU usage is charged to the explicitly authorized G04 cumulative ledger, not reset as a new budget. G02/G03 must still be repaired and independently revalidated before G04 training.
''')
    artifact_paths = [output / name for name in ("REPORT.md", "commands.log", "assets_manifest.json", "asset_tests.json")]
    artifact_paths += [published, ROOT / "configs/assembly_v2.json", ROOT / "configs/contact_calibration_v2.json", ROOT / "assets/assumptions.md"]
    for path in (stand_dir, calibration_dir, flight_dir):
        artifact_paths += [ROOT / path / name for name in ("job.json", "stdout.log", "source_snapshot.zip")]
    artifact_paths += [ROOT / stand_dir / "episodes.csv", ROOT / stand_dir / "metrics.json", ROOT / calibration_dir / "calibration.json", ROOT / flight_dir / "metrics.json", source]
    artifact_paths = list(dict.fromkeys(artifact_paths))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    evidence = {"schema_version": 1, "goal_id": "G01", "task_id": "asset_audit", "evidence_kind": "observed",
        "contract_sha256": digest(ROOT / "configs/acceptance.json"), "metrics": metrics,
        "run": {"command": "scripts/finalize_g01_reaudit.py; native commands in reports/G01/commands.log",
                "git_commit": git.stdout.strip() if git.returncode == 0 else "uncommitted-worktree-source-snapshots-attached",
                "started_at": min(j["started_at"] for j in jobs), "finished_at": max(j["finished_at"] for j in jobs),
                "gpu_hours": sum(j["charged_gpu_hours"] for j in jobs), "env_steps": sum(j["charged_env_steps"] for j in jobs)},
        "artifacts": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size} for p in artifact_paths]}
    temporary = output / "evidence.json.tmp"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n")
    temporary.replace(output / "evidence.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
