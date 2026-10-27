from __future__ import annotations

import json

import numpy as np
import pytest

from so101_capstone.policies.so101_sim_calibration import (
    SIM_CALIBRATION_FILE,
    SIM_POLICY_JOINT_NAMES,
    gazebo_positions_to_policy,
    policy_positions_to_gazebo,
)


EXPECTED_LOWER_RAD = np.array(
    [
        -1.9155686672338563,
        -1.7441249748104535,
        -1.6883093453316749,
        -1.653769279434707,
        -2.741842441713012,
        -0.17336355460059674,
    ],
    dtype=float,
)

EXPECTED_UPPER_RAD = np.array(
    [
        1.9155686672338563,
        1.7453292519943295,
        1.6883093453316749,
        1.653769279434707,
        2.838743121783737,
        1.7441249748104535,
    ],
    dtype=float,
)

POLICY_LOWER = np.array(
    [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0],
    dtype=float,
)

POLICY_UPPER = np.array(
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    dtype=float,
)


def test_simulation_calibration_file_exists_and_has_joint_order():
    payload = json.loads(
        SIM_CALIBRATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == 1
    assert tuple(payload["joint_order"]) == tuple(
        SIM_POLICY_JOINT_NAMES
    )
    assert tuple(payload["joints"]) == tuple(
        SIM_POLICY_JOINT_NAMES
    )


def test_policy_endpoints_map_to_measured_gazebo_ranges():
    assert policy_positions_to_gazebo(
        POLICY_LOWER
    ) == pytest.approx(EXPECTED_LOWER_RAD)

    assert policy_positions_to_gazebo(
        POLICY_UPPER
    ) == pytest.approx(EXPECTED_UPPER_RAD)


def test_measured_gazebo_endpoints_map_back_to_policy_ranges():
    assert gazebo_positions_to_policy(
        EXPECTED_LOWER_RAD
    ) == pytest.approx(POLICY_LOWER)

    assert gazebo_positions_to_policy(
        EXPECTED_UPPER_RAD
    ) == pytest.approx(POLICY_UPPER)
