import subprocess
import sys

import numpy as np
import pytest

from so101_capstone.environments.state_cache import (
    MEASURED_JOINT_NAMES,
)
from so101_capstone.policies import (
    SIM_JOINT_CALIBRATIONS,
    SIM_POLICY_JOINT_COUNT,
    SIM_POLICY_JOINT_NAMES,
    SimJointCalibration,
    gazebo_positions_to_policy,
    policy_positions_to_gazebo,
)


@pytest.mark.parametrize("import_statement", [
    "from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS",
    "from so101_capstone.policies import SIM_JOINT_CALIBRATIONS",
])
def test_calibration_import_does_not_load_act_or_torch(import_statement):
    code = "\n".join((
        "import sys",
        import_statement,
        "assert len(SIM_JOINT_CALIBRATIONS) == 6",
        "assert 'so101_capstone.policies.act_checkpoint_loader' not in sys.modules",
        "assert 'so101_capstone.policies.act_policy_adapter' not in sys.modules",
        "assert 'torch' not in sys.modules",
        "assert 'draccus' not in sys.modules",
        "import so101_capstone.policies as policies",
        "assert 'load_act_checkpoint' in dir(policies)",
        "assert policies.SIM_JOINT_CALIBRATIONS is SIM_JOINT_CALIBRATIONS",
    ))
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def calibration_values(attribute):
    return np.array(
        [
            getattr(calibration, attribute)
            for calibration in SIM_JOINT_CALIBRATIONS
        ],
        dtype=float,
    )


def test_simulation_calibration_matches_controller_order():
    assert SIM_POLICY_JOINT_COUNT == 6

    assert SIM_POLICY_JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    assert SIM_POLICY_JOINT_NAMES == MEASURED_JOINT_NAMES

    assert tuple(
        calibration.name
        for calibration in SIM_JOINT_CALIBRATIONS
    ) == SIM_POLICY_JOINT_NAMES


def test_calibration_uses_measured_simulation_limits():
    np.testing.assert_allclose(
        calibration_values("gazebo_lower_rad"),
        [
            -1.9155686672338563,
            -1.7441249748104535,
            -1.6883093453316749,
            -1.653769279434707,
            -2.741842441713012,
            -0.17336355460059674,
        ],
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        calibration_values("gazebo_upper_rad"),
        [
            1.9155686672338563,
            1.7453292519943295,
            1.6883093453316749,
            1.653769279434707,
            2.838743121783737,
            1.7441249748104535,
        ],
        rtol=0.0,
        atol=1e-12,
    )


def test_gazebo_endpoints_map_to_policy_endpoints():
    gazebo_lower = calibration_values(
        "gazebo_lower_rad"
    )
    gazebo_upper = calibration_values(
        "gazebo_upper_rad"
    )

    np.testing.assert_allclose(
        gazebo_positions_to_policy(gazebo_lower),
        [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        gazebo_positions_to_policy(gazebo_upper),
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_policy_endpoints_map_to_gazebo_endpoints():
    expected_lower = calibration_values(
        "gazebo_lower_rad"
    )
    expected_upper = calibration_values(
        "gazebo_upper_rad"
    )

    np.testing.assert_allclose(
        policy_positions_to_gazebo(
            [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0]
        ),
        expected_lower,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        policy_positions_to_gazebo(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        ),
        expected_upper,
        rtol=0.0,
        atol=1e-12,
    )


def test_midpoints_map_to_policy_midpoints():
    gazebo_midpoints = (
        calibration_values("gazebo_lower_rad")
        + calibration_values("gazebo_upper_rad")
    ) / 2.0

    np.testing.assert_allclose(
        gazebo_positions_to_policy(gazebo_midpoints),
        [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_round_trip_preserves_in_range_positions():
    gazebo_positions = np.array(
        [
            -1.0,
            0.25,
            1.1,
            -0.5,
            2.0,
            0.7,
        ],
        dtype=float,
    )

    policy_positions = gazebo_positions_to_policy(
        gazebo_positions
    )
    recovered = policy_positions_to_gazebo(
        policy_positions
    )

    np.testing.assert_allclose(
        recovered,
        gazebo_positions,
        rtol=0.0,
        atol=1e-12,
    )


def test_values_are_clipped_to_calibrated_ranges():
    policy_result = gazebo_positions_to_policy(
        [
            -10.0,
            10.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
        ]
    )

    np.testing.assert_allclose(
        policy_result,
        [-100.0, 100.0, -100.0, 100.0, -100.0, 100.0],
        rtol=0.0,
        atol=1e-12,
    )

    gazebo_result = policy_positions_to_gazebo(
        [
            -200.0,
            200.0,
            -200.0,
            200.0,
            -200.0,
            200.0,
        ]
    )

    expected = np.array(
        [
            SIM_JOINT_CALIBRATIONS[0].gazebo_lower_rad,
            SIM_JOINT_CALIBRATIONS[1].gazebo_upper_rad,
            SIM_JOINT_CALIBRATIONS[2].gazebo_lower_rad,
            SIM_JOINT_CALIBRATIONS[3].gazebo_upper_rad,
            SIM_JOINT_CALIBRATIONS[4].gazebo_lower_rad,
            SIM_JOINT_CALIBRATIONS[5].gazebo_upper_rad,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        gazebo_result,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_radians_is_not_a_special_home_mapping():
    zero_positions = np.zeros(6, dtype=float)

    mapped = gazebo_positions_to_policy(
        zero_positions
    )

    expected = np.array(
        [
            calibration.gazebo_to_policy(0.0)
            for calibration in SIM_JOINT_CALIBRATIONS
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        mapped,
        expected,
        rtol=0.0,
        atol=1e-12,
    )

    # Only exactly symmetric bipolar ranges map zero radians
    # to exactly zero policy units.
    assert mapped[0] == pytest.approx(0.0)
    assert mapped[1] == pytest.approx(
        -0.0345119066077757
    )
    assert mapped[2] == pytest.approx(0.0)
    assert mapped[3] == pytest.approx(0.0)

    # wrist_roll has asymmetric measured limits.
    assert mapped[4] == pytest.approx(
        -1.7363891112890286
    )

    # gripper uses a unipolar policy range [0, 100].
    assert mapped[5] == pytest.approx(
        9.041178183936504
    )


def test_conversion_results_are_read_only():
    policy_result = gazebo_positions_to_policy(
        np.zeros(6)
    )
    gazebo_result = policy_positions_to_gazebo(
        np.zeros(6)
    )

    with pytest.raises(ValueError):
        policy_result[0] = 1.0

    with pytest.raises(ValueError):
        gazebo_result[0] = 1.0


@pytest.mark.parametrize(
    "function",
    [
        gazebo_positions_to_policy,
        policy_positions_to_gazebo,
    ],
)
def test_invalid_shape_and_nonfinite_values_are_rejected(
    function,
):
    with pytest.raises(
        ValueError,
        match=r"shape \(6,\)",
    ):
        function([0.0, 0.0])

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        function(
            [
                0.0,
                0.0,
                np.nan,
                0.0,
                0.0,
                0.0,
            ]
        )


def test_invalid_joint_calibration_is_rejected():
    with pytest.raises(
        ValueError,
        match="Gazebo upper limit",
    ):
        SimJointCalibration(
            name="invalid",
            gazebo_lower_rad=1.0,
            gazebo_upper_rad=1.0,
            policy_lower=-100.0,
            policy_upper=100.0,
        )

    with pytest.raises(
        ValueError,
        match="Policy upper limit",
    ):
        SimJointCalibration(
            name="invalid",
            gazebo_lower_rad=-1.0,
            gazebo_upper_rad=1.0,
            policy_lower=100.0,
            policy_upper=100.0,
        )
