import json
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.environments import (
    PolicyObservation,
    parse_policy_action,
)
from so101_capstone.policies import (
    POLICY_EPISODE_NPZ_SCHEMA_VERSION,
    PolicyEpisodeNpzSink,
    PolicyEpisodeResult,
    PolicyTransition,
    write_policy_episode_npz,
)


def observation(value):
    return PolicyObservation(
        joint_positions=np.full(6, value, dtype=float),
        joint_velocities=np.full(
            6,
            value + 0.5,
            dtype=float,
        ),
        top_image=np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        ),
        gripper_image=np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        ),
    )


def episode_result():
    first = observation(0)
    second = observation(1)
    final = observation(2)

    first_transition = PolicyTransition(
        step_index=0,
        observation=first,
        action=parse_policy_action(np.full(6, 0.1)),
        reward=1.25,
        next_observation=second,
        terminated=False,
        truncated=False,
        info={
            "episode_step": 1,
            "target_names": (
                "target_cube_0001",
                "target_cube_0002",
            ),
        },
    )
    final_transition = PolicyTransition(
        step_index=1,
        observation=second,
        action=parse_policy_action(np.full(6, 0.2)),
        reward=2.5,
        next_observation=final,
        terminated=True,
        truncated=False,
        info={
            "episode_step": np.int64(2),
            "target_active": True,
        },
    )

    return PolicyEpisodeResult(
        initial_observation=first,
        initial_info={
            "episode_step": 0,
            "path": Path("/tmp/example"),
        },
        transitions=(
            first_transition,
            final_transition,
        ),
        final_observation=final,
        final_info=final_transition.info,
        terminated=True,
        truncated=False,
        reached_max_steps=False,
        total_reward=3.75,
    )


def test_writer_excludes_images_and_loads_without_pickle(
    tmp_path,
):
    episode = episode_result()
    output = tmp_path / "nested" / "episode.npz"

    written = write_policy_episode_npz(
        episode,
        output,
        metadata={
            "checkpoint": Path("/models/act"),
            "device": "cuda:0",
        },
    )

    assert written == output.resolve()

    with np.load(written, allow_pickle=False) as archive:
        assert int(archive["schema_version"]) == (
            POLICY_EPISODE_NPZ_SCHEMA_VERSION
        )
        np.testing.assert_array_equal(
            archive["step_index"],
            [0, 1],
        )
        np.testing.assert_allclose(
            archive["action_joint_targets"],
            [
                np.full(6, 0.1),
                np.full(6, 0.2),
            ],
        )
        np.testing.assert_allclose(
            archive["reward"],
            [1.25, 2.5],
        )
        np.testing.assert_array_equal(
            archive["done"],
            [False, True],
        )
        assert not any(
            "image" in key
            for key in archive.files
        )
        assert json.loads(
            str(archive["metadata_json"])
        ) == {
            "checkpoint": "/models/act",
            "device": "cuda:0",
        }
        transition_info = [
            json.loads(item)
            for item in archive[
                "transition_info_json"
            ].tolist()
        ]
        assert transition_info[0][
            "target_names"
        ] == [
            "target_cube_0001",
            "target_cube_0002",
        ]


def test_npz_sink_finalizes_exact_episode(
    tmp_path,
):
    episode = episode_result()
    sink = PolicyEpisodeNpzSink(
        tmp_path / "episode.npz",
        metadata={"source": "unit-test"},
    )
    sink.reset()

    for transition in episode.transitions:
        sink.append(transition)

    sink.finalize(episode)

    assert sink.written_path == (
        tmp_path / "episode.npz"
    ).resolve()
    assert sink.written_path.is_file()

    with pytest.raises(
        RuntimeError,
        match="already finalized",
    ):
        sink.finalize(episode)

    with pytest.raises(
        RuntimeError,
        match="after NPZ episode finalization",
    ):
        sink.append(episode.transitions[-1])


def test_npz_sink_rejects_different_transition_objects(
    tmp_path,
):
    episode = episode_result()
    sink = PolicyEpisodeNpzSink(
        tmp_path / "episode.npz"
    )
    sink.reset()

    replacement = PolicyTransition(
        step_index=0,
        observation=episode.initial_observation,
        action=parse_policy_action(np.full(6, 0.1)),
        reward=1.25,
        next_observation=(
            episode.transitions[0].next_observation
        ),
        terminated=False,
        truncated=False,
        info=episode.transitions[0].info,
    )

    sink.append(replacement)
    sink.append(episode.transitions[1])

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        sink.finalize(episode)


@pytest.mark.parametrize(
    "filename",
    ["episode", "episode.npy", "episode.json"],
)
def test_writer_requires_npz_suffix(
    tmp_path,
    filename,
):
    with pytest.raises(
        ValueError,
        match=r"\.npz",
    ):
        write_policy_episode_npz(
            episode_result(),
            tmp_path / filename,
        )

def test_npz_json_metadata_supports_nested_dataclasses():
    from dataclasses import dataclass
    import json

    from so101_capstone.policies.policy_episode_npz import (
        _json_text,
    )

    @dataclass(frozen=True)
    class ExamplePrivilegedState:
        target_visible: bool
        target_position: tuple[float, float, float]

    encoded = _json_text(
        {
            "privileged_state": ExamplePrivilegedState(
                target_visible=True,
                target_position=(0.1, -0.2, 0.768),
            )
        }
    )

    assert json.loads(encoded) == {
        "privileged_state": {
            "target_visible": True,
            "target_position": [0.1, -0.2, 0.768],
        }
    }
