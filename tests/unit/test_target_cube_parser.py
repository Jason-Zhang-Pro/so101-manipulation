import json

import numpy as np
import pytest

from so101_capstone.environments import (
    TargetCubeStateCache,
    parse_target_cube_states_json,
    select_target_cube,
    update_target_cube_cache_from_json,
)


def cube_entry(
    name="target_cube_0001",
    entity_id=101,
    position=None,
    orientation=None,
):
    return {
        "name": name,
        "entity_id": entity_id,
        "position": (
            {
                "x": 0.25,
                "y": 0.05,
                "z": 0.765,
            }
            if position is None
            else position
        ),
        "orientation": (
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0,
            }
            if orientation is None
            else orientation
        ),
    }


def states_payload(cubes, cube_count=None, frame_id="world"):
    return json.dumps(
        {
            "frame_id": frame_id,
            "cube_count": (
                len(cubes)
                if cube_count is None
                else cube_count
            ),
            "cubes": cubes,
        }
    )


def test_parse_valid_single_cube_message():
    parsed = parse_target_cube_states_json(
        states_payload([cube_entry()])
    )

    assert parsed.frame_id == "world"
    assert parsed.cube_count == 1

    cube = parsed.cubes[0]

    assert cube.name == "target_cube_0001"
    assert cube.entity_id == 101

    np.testing.assert_allclose(
        cube.position,
        [0.25, 0.05, 0.765],
        atol=0.0,
    )
    np.testing.assert_allclose(
        cube.quaternion_xyzw,
        [0.0, 0.0, 0.0, 1.0],
        atol=0.0,
    )


def test_named_cube_can_be_selected_from_multiple_cubes():
    parsed = parse_target_cube_states_json(
        states_payload(
            [
                cube_entry(
                    name="target_cube_0001",
                    entity_id=101,
                ),
                cube_entry(
                    name="target_cube_0002",
                    entity_id=102,
                ),
            ]
        )
    )

    selected = select_target_cube(
        parsed,
        target_name="target_cube_0002",
    )

    assert selected.name == "target_cube_0002"
    assert selected.entity_id == 102


def test_cache_update_records_receive_time_and_normalizes_quaternion():
    cache = TargetCubeStateCache()

    payload = states_payload(
        [
            cube_entry(
                orientation={
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 2.0,
                }
            )
        ]
    )

    snapshot = update_target_cube_cache_from_json(
        cache,
        payload,
        received_monotonic_s=25.0,
    )

    assert snapshot.name == "target_cube_0001"
    assert snapshot.received_monotonic_s == 25.0

    np.testing.assert_allclose(
        snapshot.quaternion_xyzw,
        [0.0, 0.0, 0.0, 1.0],
        atol=0.0,
    )


def test_multiple_cubes_require_explicit_target_name():
    parsed = parse_target_cube_states_json(
        states_payload(
            [
                cube_entry(name="target_cube_0001"),
                cube_entry(name="target_cube_0002"),
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="Exactly one",
    ):
        select_target_cube(parsed)


def test_missing_named_cube_is_rejected():
    parsed = parse_target_cube_states_json(
        states_payload([cube_entry()])
    )

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        select_target_cube(
            parsed,
            target_name="target_cube_9999",
        )


def test_invalid_json_frame_and_count_are_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_target_cube_states_json("{invalid")

    with pytest.raises(ValueError, match="frame_id"):
        parse_target_cube_states_json(
            states_payload(
                [cube_entry()],
                frame_id="base_link",
            )
        )

    with pytest.raises(ValueError, match="does not match"):
        parse_target_cube_states_json(
            states_payload(
                [cube_entry()],
                cube_count=2,
            )
        )


def test_duplicate_cube_names_are_rejected():
    payload = states_payload(
        [
            cube_entry(
                name="target_cube_0001",
                entity_id=101,
            ),
            cube_entry(
                name="target_cube_0001",
                entity_id=102,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        parse_target_cube_states_json(payload)


def test_invalid_cube_fields_are_rejected():
    invalid_position = cube_entry(
        position={
            "x": np.nan,
            "y": 0.0,
            "z": 0.765,
        }
    )

    with pytest.raises(ValueError, match="finite"):
        parse_target_cube_states_json(
            states_payload([invalid_position])
        )

    zero_quaternion = cube_entry(
        orientation={
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 0.0,
        }
    )

    with pytest.raises(ValueError, match="non-zero"):
        parse_target_cube_states_json(
            states_payload([zero_quaternion])
        )


def test_parser_preserves_source_simulation_metadata():
    import json

    from so101_capstone.environments import (
        parse_target_cube_states_json,
    )

    parsed = parse_target_cube_states_json(
        json.dumps(
            {
                "frame_id": "world",
                "sequence": 123,
                "sim_time_s": 4.1,
                "cube_count": 0,
                "cubes": [],
            }
        )
    )

    assert parsed.source_sequence == 123
    assert parsed.source_sim_time_s == pytest.approx(4.1)


def test_parser_accepts_legacy_payload_without_metadata():
    import json

    from so101_capstone.environments import (
        parse_target_cube_states_json,
    )

    parsed = parse_target_cube_states_json(
        json.dumps(
            {
                "frame_id": "world",
                "cube_count": 0,
                "cubes": [],
            }
        )
    )

    assert parsed.source_sequence is None
    assert parsed.source_sim_time_s is None


def test_json_cache_update_forwards_source_metadata():
    import json

    cache = TargetCubeStateCache()

    snapshot = update_target_cube_cache_from_json(
        cache,
        json.dumps(
            {
                "frame_id": "world",
                "sequence": 456,
                "sim_time_s": 8.2,
                "cube_count": 1,
                "cubes": [
                    cube_entry(
                        name="target_cube_0001"
                    )
                ],
            }
        ),
        received_monotonic_s=30.0,
    )

    assert snapshot.sequence == 1
    assert snapshot.source_sequence == 456
    assert snapshot.source_sim_time_s == pytest.approx(8.2)
