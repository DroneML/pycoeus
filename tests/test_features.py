import numpy as np
import pytest
import torch

from pycoeus.features import (
    get_features,
    extract_features,
    FeatureType,
    NUM_FLAIR_CLASSES,
    get_flair_model_file_name,
    load_model, pad, calculate_pad_sizes_1d, unpad,
)
from pycoeus.utils.models import UNet


class TestExtractFeatures:
    def test_get_identity_features(self):
        input_data = np.array(get_generated_multiband_image())
        result = get_features(input_data, None, FeatureType.IDENTITY, None)
        assert np.array_equal(result.data, input_data.data)

    @pytest.mark.parametrize(
        ["n_bands", "width", "height"],
        [
            (3, 2, 2),  # too small to be processed by the model, requires padding
            (3, 8, 8),  # too small to be processed by the model, requires padding
            (3, 16, 16),  # smallest size that can natively be processed by the model
            (3, 61, 39),  # not divisible by 16 so requires padding in both directions
            (3, 64, 48),  # smallest dimensions, > line above, that don't require padding
            (1, 512, 512),  # size of the model's training data (easiest case)
            (3, 1210, 718),  # not divisible by 16 so requires padding in both directions
        ],
    )
    def test_extract_flair_features(self, n_bands, width, height):
        input_data = np.array(get_generated_multiband_image(n_bands=n_bands, width=width, height=height))
        result = extract_features(input_data, FeatureType.FLAIR, model_scale=0.125)
        assert np.array_equal(result.shape, [n_bands * NUM_FLAIR_CLASSES] + list(input_data.shape[1:]))

    def test_extract_features_unsupported_type(self):
        input_data = np.array([[1, 2], [3, 4]])
        with pytest.raises(KeyError):
            extract_features(input_data, "UNSUPPORTED_TYPE")


def get_generated_multiband_image(n_bands=3, width=512, height=512):
    return np.random.random(size=[n_bands, width, height])


@pytest.mark.parametrize(
    ["model_scale", "file_name"],
    [
        (1.0, "flair_toy_ep15_scale1_0.pth"),
        (0.5, "flair_toy_ep15_scale0_5.pth"),
        (0.25, "flair_toy_ep15_scale0_25.pth"),
        (0.125, "flair_toy_ep15_scale0_125.pth"),
    ],
)
def test_get_flair_model_file_name_with_valid_scales(model_scale, file_name):
    assert get_flair_model_file_name(model_scale) == file_name


@pytest.mark.parametrize(["model_scale"], [(2,), (0.001,)])
def test_get_flair_model_file_name_with_invalid_scales(model_scale):
    with pytest.raises(ValueError):
        get_flair_model_file_name(model_scale)


@pytest.mark.downloader
@pytest.mark.parametrize(["model_scale", "source"], [
    (0.125, "Huggingface"),
    (0.25, "Huggingface"),
    (0.5, "Huggingface"),
    (1.0, "Huggingface"),
    (0.25, "Surfdrive"),
    (1.0, "Surfdrive"),
    (1.0, None),
])
def test_load_model_downloads(model_scale, source, tmpdir):
    sources = (source,) if source is not None else None
    model, device = load_model(model_scale, models_dir=tmpdir, sources=sources)
    assert model is not None
    assert type(model) == UNet



def test_pad():
    """Test padding of input band to ensure dimensions are divisible by 16."""
    input_band = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]])[None, None, :, :], dtype=torch.float32)
    expected_output = torch.tensor(
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])[None, None, :, :], dtype=torch.float32
    )

    padded_band = pad(input_band, band_name="test_band")

    assert torch.equal(padded_band, expected_output)


@pytest.mark.parametrize(["input_size"], [((1, 1, 1, 2),), ((1, 1, 17, 23),), ((1, 1, 123, 196),)])
def test_pad_unpad(input_size):
    rng = np.random.default_rng(0)
    input_band = torch.tensor(rng.random(input_size), dtype=torch.float32)

    padded_band = pad(input_band, band_name=f"{input_size}")
    unpadded_band = unpad(padded_band, input_size)

    assert torch.equal(input_band, unpadded_band)


@pytest.mark.parametrize("dim_size, expected_pad_sizes", [
    (1, (7, 8)),
    (4, (6, 6)),
    (15, (0, 1)),
    (16, (0, 0)),
    (17, (7, 8)),
    (31, (0, 1)),
    (32, (0, 0))
])
def test_calculate_pad_size_1d(dim_size, expected_pad_sizes):
    """Test calculation of padding sizes to make a dimension size divisible by 16."""
    pad_sizes = calculate_pad_sizes_1d(dim_size)
    assert pad_sizes == expected_pad_sizes
