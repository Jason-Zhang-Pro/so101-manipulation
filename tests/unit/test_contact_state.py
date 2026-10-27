from types import SimpleNamespace

import pytest

from so101_capstone.environments.contact_state import FingerContactCache


def update(cache, t, left=False, right=False, target="target_cube_1"):
    for side, touching in (("left", left), ("right", right)):
        cache.update(side, sim_time_s=t, collision_pairs=(("robot::finger::collision", f"{target}::cube_link::cube_collision"),) if touching else ())


def test_empty_contact_message_is_valid_but_missing_stream_is_unknown():
    cache = FingerContactCache()
    assert not cache.evidence(target_name="target_cube_1", end_time_s=0).valid
    update(cache, 0)
    result = cache.evidence(target_name="target_cube_1", end_time_s=0)
    assert result.valid and not result.left_contact and not result.right_contact


def test_sequential_finger_touches_do_not_certify_bilateral_grasp():
    cache = FingerContactCache()
    update(cache, 0, left=True)
    update(cache, .05, right=True)
    result = cache.evidence(target_name="target_cube_1", start_time_s=0, end_time_s=.05)
    assert result.valid and not result.contacts_simultaneous
    assert result.simultaneous_duration_s == 0


def test_bilateral_overlap_is_target_specific_and_time_bounded():
    cache = FingerContactCache()
    update(cache, 0, True, True)
    update(cache, .05, True, True)
    result = cache.evidence(target_name="target_cube_1", start_time_s=0, end_time_s=.05)
    assert result.contacts_simultaneous
    assert result.simultaneous_duration_s == pytest.approx(.05)
    assert not cache.evidence(target_name="target_cube_10", end_time_s=.05).left_contact
    assert not cache.evidence(target_name="target_cube_1", end_time_s=.06).valid


def test_missing_interval_never_extrapolates_or_fills_stability():
    cache = FingerContactCache()
    update(cache, 0, True, True)
    update(cache, 1, True, True)
    assert not cache.evidence(target_name="target_cube_1", start_time_s=0, end_time_s=1).valid


def test_clock_reset_clears_both_channels():
    cache = FingerContactCache()
    update(cache, 1, True, True)
    cache.update("left", sim_time_s=0)
    assert not cache.evidence(target_name="target_cube_1", end_time_s=0).valid
    assert not cache.history()["right"]


def test_ros_like_message_preserves_timestamp_and_collision_identity():
    cache = FingerContactCache()
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=5_000_000)), contacts=[
        SimpleNamespace(collision1=SimpleNamespace(name="finger"), collision2=SimpleNamespace(name="world::target_cube_1::cube::collision"))])
    cache.update_from_message("left", message)
    item = cache.history()["left"][-1]
    assert item.sim_time_s == pytest.approx(2.005)
    assert item.touches("target_cube_1")
    with pytest.raises(ValueError):
        cache.update_from_message("right", SimpleNamespace(contacts=[]))
