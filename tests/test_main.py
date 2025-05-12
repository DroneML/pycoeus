import shutil
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import rioxarray
import xarray as xr
from scipy.spatial.distance import dice

from pycoeus.features import FeatureType, get_features_path
from pycoeus.main import read_input_and_labels_and_save_predictions, prepare_training_data, make_predictions
from pycoeus.utils.datasets import normalize_single_band
from pycoeus.utils.geospatial import get_label_array
from .test_cases import TestCase, test_case1210, test_case512
from .utils import TEST_DATA_FOLDER

rng = np.random.default_rng(4242)


@pytest.mark.parametrize(
    "test_case, feature_type, model_scale, dice_similarity_threshold, compute_mode",
    [
        pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "normal", marks=pytest.mark.slow),
        pytest.param(
            test_case1210, FeatureType.FLAIR, 0.125, None, "normal"
        ),  # also slow, but necessary to test on each run
        pytest.param(test_case512, FeatureType.IDENTITY, None, 0.90, "normal", marks=pytest.mark.slow),
        pytest.param(test_case512, FeatureType.FLAIR, 0.125, 0.82, "normal", marks=pytest.mark.slow),
        pytest.param(test_case512, FeatureType.FLAIR, 1.0, 0.95, "normal", marks=pytest.mark.slow),
        pytest.param(test_case512, FeatureType.FLAIR, 1.0, 0.95, "parallel", marks=pytest.mark.slow),
        pytest.param(test_case512, FeatureType.FLAIR, 1.0, 0.95, "safe", marks=pytest.mark.slow),
        pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "parallel", marks=pytest.mark.slow),
        pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "parallel", marks=pytest.mark.slow),
        pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "safe", marks=pytest.mark.slow),
        pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "safe", marks=pytest.mark.slow),
    ],
    ids=lambda e: str(e),
)
def test_integration(tmpdir, test_case: TestCase, feature_type, model_scale, dice_similarity_threshold, compute_mode):
    tmpdir = Path(tmpdir)
    np.random.seed(0)

    input_path = copy_file_and_get_new_path(test_case.image_filename, tmpdir)
    labels_pos_path = copy_file_and_get_new_path(test_case.labels_pos_filename, tmpdir)
    labels_neg_path = copy_file_and_get_new_path(test_case.labels_neg_filename, tmpdir)
    predictions_path = (
        Path(tmpdir) / f"{test_case.image_filename}_predictions_{str(feature_type)}_model_{model_scale}.tif"
    )
    classifier_path = (Path(tmpdir) / f"{test_case512.image_filename}_classifier.pkl")

    read_input_and_labels_and_save_predictions(
        input_path,
        labels_pos_path,
        labels_neg_path,
        classifier_path,
        predictions_path,
        feature_type=feature_type,
        model_scale=model_scale,
    )

    assert predictions_path.exists()
    assert classifier_path.exists()

    # Check DICE similarity if threshold is provided
    if dice_similarity_threshold is None and test_case.ground_truth_filename is None:
        return

    truth = rioxarray.open_rasterio(TEST_DATA_FOLDER / "test_image_512x512_out_ground_truth.tif").astype(np.int16)
    predictions = rioxarray.open_rasterio(predictions_path).astype(np.float64)
    dice_similarity = 1 - dice(truth.data.flatten() > 0.5, predictions.data.flatten() > 0.999)
    print(f"DICE similarity index: {dice_similarity}")
    for t in range(0, 100):
        dice_similarity = 1 - dice(truth.data.flatten() > 0.5, predictions.data.flatten() > t / 100)
        print(f"DICE similarity index ({t / 100}): {dice_similarity}")
    assert dice_similarity > dice_similarity_threshold


def copy_file_and_get_new_path(test_image, tmpdir):
    input_path = Path(tmpdir) / test_image
    shutil.copy(TEST_DATA_FOLDER / test_image, input_path)
    return input_path


@pytest.mark.parametrize(
    "input_path, feature_type, expected_path",
    [
        ("input.tiff", FeatureType.FLAIR, "input_FLAIR.tiff"),
        ("../path/to/input.tiff", FeatureType.FLAIR, "../path/to/input_FLAIR.tiff"),
        ("../path/to/input.tiff", FeatureType.IDENTITY, "../path/to/input.tiff"),
    ],
)
def test_get_features_path(input_path, feature_type, expected_path):
    features_path = get_features_path(Path(input_path), feature_type)
    assert features_path == Path(expected_path)


@pytest.mark.parametrize("array_type", ["numpy", "dask"])
def test_prepare_training_data(array_type):
    random = np.random.default_rng(0)
    length = 200
    random_data = random.integers(low=0, high=256, size=(5, length, length))
    labels = random.choice([0, 1, 2], size=(1, length, length), replace=True)
    if array_type == "numpy":
        input_data = random_data
    elif array_type == "dask":
        input_data = da.from_array(random_data)

    prepare_training_data(input_data, labels)


@pytest.mark.parametrize(
    "label_options",
    [
        [1, 2],  # no negative labels
        [0, 2],  # no positive labels
    ],
)
def test_zero_positive_labels_raises_value_error(label_options):
    print(f"{label_options=}")
    rng = np.random.default_rng(0)
    length = 200
    random_data = rng.integers(low=0, high=256, size=(5, length, length))
    labels = rng.choice(label_options, size=(1, length, length), replace=True)

    with pytest.raises(ValueError):
        prepare_training_data(random_data, labels)


@pytest.mark.parametrize(
    "nparr",
    [
        rng.integers(low=0, high=255, size=(1, 5, 7)),
        rng.integers(low=0, high=255, size=(3, 10, 29)),
        rng.random((1, 10, 29)),
        rng.random((1, 50, 22)),
    ],
)
def test_normalization(nparr):
    # numpy array and dask array should give the same results

    # Create a dask array from the numpy array
    da_arr = da.from_array(nparr, chunks=(1, 5, 5))

    # Covert both to xarray DataArray to make like a raster
    raster_np = xr.DataArray(
        nparr,
        dims=["band", "y", "x"],
        coords={"band": np.arange(nparr.shape[0]), "y": np.arange(nparr.shape[1]), "x": np.arange(nparr.shape[2])},
    )

    raster_da = xr.DataArray(
        da_arr,
        dims=["band", "y", "x"],
        coords={"band": np.arange(da_arr.shape[0]), "y": np.arange(da_arr.shape[1]), "x": np.arange(da_arr.shape[2])},
    )

    np_results = xr.apply_ufunc(
        normalize_single_band,
        raster_np,
        input_core_dims=[["band"]],
        output_core_dims=[["band"]],
        dask="allowed",
    ).transpose(*raster_np.dims)

    da_results = xr.apply_ufunc(
        normalize_single_band,
        raster_da,
        input_core_dims=[["band"]],
        output_core_dims=[["band"]],
        dask="allowed",
    ).transpose(*raster_da.dims)

    # Check that the results are almost equal
    assert np.allclose(np_results.values, da_results.values, rtol=1e-8, atol=1e-8)


def train_predict_score(features, raster, tmpdir, use_case):
    pos_gdf = gpd.read_file(TEST_DATA_FOLDER / use_case.labels_pos_filename).to_crs(raster.rio.crs)
    neg_gdf = gpd.read_file(TEST_DATA_FOLDER / use_case.labels_neg_filename).to_crs(raster.rio.crs)
    predictions_path = Path(tmpdir) / "predictions.tif"
    # Get label arrays
    labels = get_label_array(features, pos_gdf, neg_gdf, compute_mode="normal")
    # Make predictions
    prediction_map = make_predictions(features.data, labels.data)
    # Use raster as the template and assign data
    prediction_raster = raster.isel(band=0).drop_vars(["band"]).expand_dims(band=prediction_map.shape[0])
    prediction_raster.data = prediction_map
    # Save predictions
    prediction_raster.rio.to_raster(predictions_path)
    truth = rioxarray.open_rasterio(TEST_DATA_FOLDER / use_case.ground_truth_filename).astype(np.int16)
    predictions = rioxarray.open_rasterio(predictions_path).astype(np.int16)
    dice_similarity = 1 - dice(truth.data.flatten(), predictions.data.flatten())
    return dice_similarity
