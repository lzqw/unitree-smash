"""CPU kinematics of the pinned standard PiPER and declared racket assembly.

This module does not simulate dynamics, self collision, or actuator tracking.
Its optimized poses are search seeds, never executable-trajectory evidence.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


class PiperGeometry:
    def __init__(self, root: Path | None = None):
        if root is None:
            configured = os.environ.get("QUADSMASH_G07_EXECUTION_ROOT")
            root = Path(configured).expanduser().resolve() if configured else ROOT
        root = Path(root).resolve()
        self.assembly = json.loads((root / "configs/assembly_v2.json").read_text())
        self.urdf = root / self.assembly["piper"]["source"]
        tree = ET.parse(self.urdf).getroot()
        self.joints = []
        bounds = []
        for index in range(1, 7):
            joint = tree.find(f"joint[@name='joint{index}']")
            origin = joint.find("origin")
            position = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            self.joints.append((position, Rotation.from_euler("xyz", rpy).as_matrix()))
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            if not np.array_equal(axis, [0, 0, 1]):
                raise ValueError("Pinned arm joint axes changed; revalidate kinematics")
            limit = joint.find("limit")
            bounds.append((float(limit.get("lower")), float(limit.get("upper"))))
        self.bounds = np.asarray(bounds)
        mount = self.assembly["mount"]
        self.arm_origin = (np.array(mount["plate_center_relative_to_go2_base_m"])
                           + np.array(mount["arm_base_relative_to_plate_m"]))
        self.tool_offset = np.array(self.assembly["racket"]["blade_center_relative_to_link6_m"])
        self.half_extents = np.array(self.assembly["racket"]["blade_size_m"]) / 2

    def forward(self, q, base_position=(0., 0., .323), base_rpy=(0., 0., 0.)):
        """Return racket-center world position and local-to-world rotation."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint positions")
        rotation = Rotation.from_euler("xyz", base_rpy).as_matrix()
        position = np.asarray(base_position, dtype=float) + rotation @ self.arm_origin
        for angle, (offset, origin_rotation) in zip(q, self.joints):
            position = position + rotation @ offset
            rotation = rotation @ origin_rotation @ Rotation.from_rotvec([0, 0, angle]).as_matrix()
        return position + rotation @ self.tool_offset, rotation

    def contact_center(self, q, uv=(0., 0.), face=1, radius=.025, **kwargs):
        """Sphere center at an interior flat-face contact, not a box corner."""
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (2,) or np.any(np.abs(uv) > self.half_extents[1:]):
            raise ValueError("Contact point is outside the effective square face")
        if face not in (-1, 1) or radius < 0:
            raise ValueError("Invalid face or sphere radius")
        position, rotation = self.forward(q, **kwargs)
        offset = np.r_[face * (self.half_extents[0] + radius), uv]
        return position + rotation @ offset, face * rotation[:, 0]

    def vertical_support(self, rotation):
        """Exact box vertical half extension sum_i |R_zi| h_i."""
        return float(np.abs(np.asarray(rotation)[2]) @ self.half_extents)

    def vertices(self, q, **kwargs):
        position, rotation = self.forward(q, **kwargs)
        local = np.array(list(itertools.product((-1, 1), repeat=3))) * self.half_extents
        return position + local @ rotation.T

    def contact_jacobian(self, q, uv=(0., 0.), face=1, **kwargs):
        """Finite-difference world translational Jacobian of a face point."""
        delta = 1e-6
        q = np.asarray(q, dtype=float)
        columns = []
        for axis in np.eye(6):
            plus, _ = self.contact_center(q + delta * axis, uv, face, **kwargs)
            minus, _ = self.contact_center(q - delta * axis, uv, face, **kwargs)
            columns.append((plus - minus) / (2 * delta))
        return np.stack(columns, axis=1)

    def unrestricted_reach_bound(self, radius=.025):
        """Triangle-inequality distance bound from the Go2 root frame.

        This is loose but independent of sampling and root orientation. World
        height still requires a separately justified bound on root height.
        """
        return float(np.linalg.norm(self.arm_origin)
                     + sum(np.linalg.norm(offset) for offset, _ in self.joints)
                     + np.linalg.norm(self.tool_offset)
                     + np.linalg.norm(self.half_extents) + radius)
