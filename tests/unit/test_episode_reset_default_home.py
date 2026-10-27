from __future__ import annotations

import numpy as np

from so101_capstone.environments.episode_reset import (
    EpisodeResetConfig,
)


EXPECTED_HOME_DEG = np.array(
    [
        0.0,
        -98.0,
        92.0,
        65.0,
        0.0,
        -5.0,
    ],
    dtype=float,
)


def test_default_episode_reset_home_matches_selected_pose():
    config = EpisodeResetConfig()

    actual_rad = np.asarray(
        config.home_joint_positions,
        dtype=float,
    )
    expected_rad = np.asarray(
        (0.0, -1.710422666954443, 1.605702911834783, 1.134464013796314, 0.0, -0.087266462599716),
        dtype=float,
    )

    np.testing.assert_allclose(
        actual_rad,
        expected_rad,
        rtol=0.0,
        atol=1e-12,
    )
