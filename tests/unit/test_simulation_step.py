import pytest

from so101_capstone.environments.simulation_step import (
    SimulationStepSchedule,
)


def test_30_hz_with_one_millisecond_physics_step():
    schedule = SimulationStepSchedule(
        control_hz=30.0,
        physics_step_s=0.001,
    )

    counts = [
        schedule.next_step_count()
        for _ in range(9)
    ]

    assert counts == [
        33, 33, 34,
        33, 33, 34,
        33, 33, 34,
    ]


def test_thirty_control_steps_advance_exactly_one_second():
    schedule = SimulationStepSchedule(
        control_hz=30.0,
        physics_step_s=0.001,
    )

    counts = [
        schedule.next_step_count()
        for _ in range(30)
    ]

    assert sum(counts) == 1000
    assert schedule.control_step_count == 30
    assert schedule.physics_step_count == 1000
    assert schedule.elapsed_sim_time_s == pytest.approx(1.0)
    assert schedule.ideal_elapsed_sim_time_s == pytest.approx(
        1.0
    )
    assert schedule.timing_error_s == pytest.approx(0.0)


def test_schedule_has_no_long_term_drift():
    schedule = SimulationStepSchedule(
        control_hz=30.0,
        physics_step_s=0.001,
    )

    for _ in range(300):
        schedule.next_step_count()

    assert schedule.physics_step_count == 10_000
    assert schedule.elapsed_sim_time_s == pytest.approx(10.0)
    assert schedule.ideal_elapsed_sim_time_s == pytest.approx(
        10.0
    )
    assert schedule.timing_error_s == pytest.approx(0.0)


def test_reset_restarts_schedule():
    schedule = SimulationStepSchedule()

    assert schedule.next_step_count() == 33
    assert schedule.next_step_count() == 33

    schedule.reset()

    assert schedule.control_step_count == 0
    assert schedule.physics_step_count == 0
    assert schedule.next_step_count() == 33


@pytest.mark.parametrize(
    ("control_hz", "physics_step_s"),
    [
        (0.0, 0.001),
        (30.0, 0.0),
        (2000.0, 0.001),
    ],
)
def test_invalid_timing_configuration_is_rejected(
    control_hz,
    physics_step_s,
):
    with pytest.raises(ValueError):
        SimulationStepSchedule(
            control_hz=control_hz,
            physics_step_s=physics_step_s,
        )
