import math

import numpy as np
import pytest

from so101_capstone.environments.top_camera_processing import (
    TopCameraImageProcessor,
    TopCameraProcessingConfig,
    default_simulation_top_camera_processing_config,
)


def small_config(**overrides):
    values = {
        "input_height": 24,
        "input_width": 32,
        "output_height": 12,
        "output_width": 16,
        "horizontal_fov_rad": math.radians(95.0),
    }
    values.update(overrides)
    return TopCameraProcessingConfig(**values)


def checkerboard(
    height=24,
    width=32,
):
    y, x = np.indices((height, width))
    pattern = ((x // 4 + y // 4) % 2) * 255

    return np.stack(
        (
            pattern,
            np.roll(pattern, 1, axis=1),
            np.roll(pattern, 1, axis=0),
        ),
        axis=2,
    ).astype(np.uint8)


def test_default_shapes_match_camera_and_policy():
    config = TopCameraProcessingConfig()

    assert config.input_shape == (600, 800, 3)
    assert config.output_shape == (480, 640, 3)


def test_identity_processing_resizes_uniform_image():
    processor = TopCameraImageProcessor(
        small_config()
    )
    image = np.full(
        (24, 32, 3),
        73,
        dtype=np.uint8,
    )

    result = processor.process(image)

    assert result.shape == (12, 16, 3)
    assert result.dtype == np.uint8
    assert np.all(result == 73)
    assert not result.flags.writeable


def test_nonzero_distortion_changes_checkerboard():
    image = checkerboard()

    identity = TopCameraImageProcessor(
        small_config()
    ).process(image)

    distorted = TopCameraImageProcessor(
        small_config(
            k1=-0.12,
            k2=0.02,
            distorted_focal_scale=1.08,
        )
    ).process(image)

    assert distorted.shape == (12, 16, 3)
    assert distorted.dtype == np.uint8
    assert not np.array_equal(
        distorted,
        identity,
    )


def test_input_image_is_not_modified():
    processor = TopCameraImageProcessor(
        small_config(
            k1=-0.08,
            distorted_focal_scale=1.05,
        )
    )
    image = checkerboard()
    original = image.copy()

    processor.process(image)

    np.testing.assert_array_equal(
        image,
        original,
    )


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((24, 32, 3), dtype=np.float32),
        np.zeros((23, 32, 3), dtype=np.uint8),
        np.zeros((24, 32), dtype=np.uint8),
        np.zeros((24, 32, 4), dtype=np.uint8),
    ],
)
def test_invalid_images_are_rejected(image):
    processor = TopCameraImageProcessor(
        small_config()
    )

    with pytest.raises(ValueError):
        processor.process(image)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_height": 0},
        {"input_width": -1},
        {"output_height": 0},
        {"horizontal_fov_rad": 0.0},
        {"horizontal_fov_rad": math.pi},
        {"k1": float("nan")},
        {"principal_x_fraction": -0.1},
        {"principal_y_fraction": 1.1},
        {"distorted_focal_scale": 0.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        small_config(**kwargs)

def test_default_simulation_profile_enables_distortion():
    config = default_simulation_top_camera_processing_config()

    assert config.input_shape == (600, 800, 3)
    assert config.output_shape == (480, 640, 3)
    assert config.horizontal_fov_rad == pytest.approx(
        math.radians(100.0)
    )
    assert config.k1 == pytest.approx(-0.226598)
    assert config.k2 == pytest.approx(0.039593)
    assert config.p1 == pytest.approx(0.0)
    assert config.p2 == pytest.approx(0.0)
    assert config.k3 == pytest.approx(0.0)
    assert config.principal_x_fraction == pytest.approx(
        0.4801
    )
    assert config.principal_y_fraction == pytest.approx(
        0.4979
    )
    assert config.distorted_focal_scale == pytest.approx(
        1.40
    )
    assert np.any(
        config.distortion_coefficients != 0.0
    )



def test_auto_crop_removes_all_remap_border_pixels():
    config = small_config(
        k1=-0.226598,
        k2=0.039593,
        distorted_focal_scale=1.0,
        crop_to_valid_region=True,
        valid_source_margin_px=1.0,
    )
    processor = TopCameraImageProcessor(config)

    # Any black output pixel would have come from BORDER_CONSTANT because
    # the complete source image is white.
    image = np.full(
        config.input_shape,
        255,
        dtype=np.uint8,
    )

    result = processor.process(image)

    assert result.shape == config.output_shape
    assert result.dtype == np.uint8
    assert np.all(result == 255)

    x0, y0, x1, y1 = (
        processor.crop_bounds_xyxy
    )
    crop_width = x1 - x0
    crop_height = y1 - y0

    assert crop_width > 0
    assert crop_height > 0
    assert (
        crop_width * config.output_height
        == crop_height * config.output_width
    )


def test_default_simulation_profile_uses_fixed_crop():
    config = (
        default_simulation_top_camera_processing_config()
    )

    assert config.crop_to_valid_region is False
    assert config.fixed_crop_xyxy == (10, 18, 758, 579)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_to_valid_region": 1},
        {"valid_source_margin_px": -0.1},
        {"valid_source_margin_px": 20.0},
    ],
)
def test_invalid_auto_crop_configuration_is_rejected(
    kwargs,
):
    with pytest.raises(ValueError):
        small_config(**kwargs)

def test_default_fixed_crop_contains_no_remap_border():
    config = (
        default_simulation_top_camera_processing_config()
    )
    processor = TopCameraImageProcessor(config)

    image = np.full(
        config.input_shape,
        255,
        dtype=np.uint8,
    )

    result = processor.process(image)

    assert processor.crop_bounds_xyxy == (10, 18, 758, 579)
    assert result.shape == config.output_shape
    assert np.all(result == 255)


def test_reduced_resolution_preserves_reference_crop_geometry():
    """Keep the verified optics while reducing raw rendering workload."""

    config = default_simulation_top_camera_processing_config()

    assert config.input_shape == (600, 800, 3)
    assert config.output_shape == (480, 640, 3)
    assert config.horizontal_fov_rad == pytest.approx(
        math.radians(100.0)
    )
    assert config.k1 == pytest.approx(-0.226598)
    assert config.k2 == pytest.approx(0.039593)
    assert config.p1 == pytest.approx(0.0)
    assert config.p2 == pytest.approx(0.0)
    assert config.k3 == pytest.approx(0.0)
    assert config.principal_x_fraction == pytest.approx(0.4801)
    assert config.principal_y_fraction == pytest.approx(0.4979)
    assert config.distorted_focal_scale == pytest.approx(1.40)

    x0, y0, x1, y1 = config.fixed_crop_xyxy
    crop_width = x1 - x0
    crop_height = y1 - y0

    assert (x0, y0, x1, y1) == (10, 18, 758, 579)
    assert (crop_width, crop_height) == (748, 561)

    # The crop remains exactly 4:3, so policy resize is geometry-neutral.
    assert crop_width * config.output_height == (
        crop_height * config.output_width
    )

    # Preserve the principal-point-centered view to within one raw pixel.
    crop_center_x = 0.5 * (x0 + x1)
    crop_center_y = 0.5 * (y0 + y1)
    expected_center_x = (
        config.input_width * config.principal_x_fraction
    )
    expected_center_y = (
        config.input_height * config.principal_y_fraction
    )

    assert abs(crop_center_x - expected_center_x) <= 1.0
    assert abs(crop_center_y - expected_center_y) <= 1.0
