import argparse
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import dask.array as da
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import rioxarray
import dask
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import logging

from pycoeus.features import get_features, FeatureType, DEFAULT_CHUNK_OVERLAP
from pycoeus.logging_config import setup_logger, log_duration, log_array
from pycoeus.utils.io import read_geotiff
from pycoeus.utils.geospatial import get_label_array
from pycoeus.utils.datasets import normalize_single_band

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


def read_input_and_labels_and_save_predictions(
    raster_path: Path,
    pos_labels_path: Path,
    neg_labels_path: Path,
    output_path: Path,
    feature_type=FeatureType.IDENTITY,
    features_path: Path = None,
    compute_mode: Literal["normal", "parallel", "safe"] = "normal",
    chunks: dict = None,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    **extractor_kwargs,
) -> None:
    logger.info("read_input_and_labels_and_save_predictions called with the following arguments:")
    for k, v in locals().items():
        logger.info(f"{k}: {v}")

    if pos_labels_path == neg_labels_path:
        raise ValueError(f'Positive and negative labels must be different files, both were set to "{pos_labels_path}".')

    # Set compute mode, and get dask kwargs for reading raster data
    dask_kwargs = _set_compute_mode(compute_mode, chunks)

    # Load raster data
    raster = read_geotiff(raster_path, **dask_kwargs)

    # Ensure the raster has correct crs
    # In QGIS environment, rasterio may not be able to read the crs correctly
    # However, osgeo.gdal, which is in QGIS, can read the crs correctly
    # Therefore, we use osgeo.gdal to read the crs, when osgeo is available
    try:
        from osgeo import gdal

        dataset = gdal.Open(raster_path)
        projection = dataset.GetProjection()
        raster = raster.rio.write_crs(projection, inplace=True)
        logger.info("Used osgeo.gdal to read the crs")
    except ImportError:
        logger.info("Used rioxarray to read the crs")

    # Normalize the raster data per band
    raster_norm = xr.apply_ufunc(
        normalize_single_band,
        raster,
        input_core_dims=[["band"]],
        output_core_dims=[["band"]],
        dask="allowed",
    ).transpose(*raster.dims)

    # Extract features
    features = get_features(
        raster_norm,
        raster_path,
        feature_type,
        features_path,
        chunk_overlap=chunk_overlap,
        compute_mode=compute_mode,
        **extractor_kwargs,
    )

    # Load vector labels as geodataframes, and align CRS with input data
    pos_gdf = gpd.read_file(pos_labels_path).to_crs(raster_norm.rio.crs)
    neg_gdf = gpd.read_file(neg_labels_path).to_crs(raster_norm.rio.crs)

    # Get label arrays
    labels = get_label_array(features, pos_gdf, neg_gdf, compute_mode=compute_mode)

    # Make predictions
    prediction_map = make_predictions(features.data, labels.data)

    # Use raster as the template and assign data
    prediction_raster = raster_norm.isel(band=0).drop_vars(["band"]).expand_dims(band=prediction_map.shape[0])
    prediction_raster.data = prediction_map

    # Save predictions
    prediction_raster.rio.to_raster(output_path)

    return output_path


def make_predictions(input_data: ndarray, labels: ndarray) -> ndarray:
    """Makes predictions by training a classifier and using it for inference.

    Expects input data with shape of [channels, width, height] and labels of shape [classes, width, height]
        :param input_data: input data with shape of [channels, width, height]
        :param labels: labels with shape [1, width, height]
    :return: probabilities with shape [class_values, width, height]
    """
    with log_duration("Prepare train data", logger):
        train_data, train_labels = prepare_training_data(input_data, labels)

    classifier = get_classifier()

    with log_duration("Train model", logger):
        classifier.fit(train_data, train_labels)

    with log_duration("Make predictions", logger):
        predictions = classifier.predict_proba(input_data.reshape((input_data.shape[0], -1)).transpose())
        prediction_map = predictions.transpose().reshape((predictions.shape[1], *input_data.shape[1:]))
        log_array(prediction_map, logger, array_name="Predictions")

    return prediction_map


class ClassifierType(Enum):
    RANDOM_FOREST = 1
    XGBOOST = 2
    MLP = 3
    SVM = 4
    LOGISTIC_REGRESSION = 5

    @staticmethod
    def from_string(s):
        try:
            return ClassifierType[s]
        except KeyError:
            raise ValueError()


def get_classifier(classifier_type=ClassifierType.RANDOM_FOREST):
    logger.info(f"Using classifier: {classifier_type.name}")
    if classifier_type == ClassifierType.RANDOM_FOREST:
        return RandomForestClassifier(n_estimators=100)
    if classifier_type == ClassifierType.XGBOOST:
        return XGBClassifier(n_estimators=5000)
    if classifier_type == ClassifierType.MLP:
        return MLPClassifier()
    if classifier_type == ClassifierType.SVM:
        return SVC(probability=True)
    if classifier_type == ClassifierType.LOGISTIC_REGRESSION:
        return LogisticRegression()

    raise ValueError(f"Invalid classifier type: {classifier_type}")


def prepare_training_data(input_data, labels):
    """
    Prepares training data for a binary classification task.

    Parameters:
    - input_data: A 3D array-like object (e.g., NumPy or Dask array) where the first dimension represents
      instances, and the last two dimensions represent spatial data.
    - labels: A 3D array-like object (e.g., NumPy or Dask array) where the first dimension is the class index
      (only single-class supported), and the last two dimensions represent spatial labels.

    Process:
    1. Flattens the label array for the first class to a 1D array.
    2. Separates the positive and negative instances from the input data based on the labels.
    3. Computes the number of labeled and unlabeled instances and logs the statistics.
    4. Concatenates positive and negative instances into training data and corresponding labels.

    Returns:
    - train_data: A 2D array where each row is a training instance and each column is a feature.
    - train_labels: A 1D array containing the labels corresponding to the rows in `train_data`.
    """
    # Reshape input data to [n_instances, n_features]
    class1_labels = labels[0]  # Only single class is supported
    flattened = class1_labels.flatten()
    positive_instances = input_data.reshape((input_data.shape[0], -1))[:, flattened == 1].transpose()
    negative_instances = input_data.reshape((input_data.shape[0], -1))[:, flattened == 0].transpose()
    n_total_instances = np.prod(labels.shape[-2:])

    _validate_and_log_instance_numbers(negative_instances.shape[0], positive_instances.shape[0], n_total_instances)
    # Subsample training data
    sampled_positive_instances = _subsample(positive_instances, 10000)
    sampled_negative_instances = _subsample(negative_instances, 10000)
    n_sampled_positive = sampled_positive_instances.shape[0]
    n_sampled_negative = sampled_negative_instances.shape[0]
    n_sampled_labeled = n_sampled_negative + n_sampled_positive
    logger.info(
        f"Training on {n_sampled_positive} ({round(100 * n_sampled_positive / n_sampled_labeled, 2)}%) positive labels and {n_sampled_negative} ({round(100 * n_sampled_negative / n_sampled_labeled, 2)}%) negative labels "
    )
    # Shuffle training data
    total_sample_size = sampled_positive_instances.shape[0] + sampled_negative_instances.shape[0]
    order = np.arange(total_sample_size)
    np.random.shuffle(order)
    train_data = np.concatenate((sampled_positive_instances, sampled_negative_instances))[order]
    train_labels = np.concatenate(
        (
            (np.ones(shape=[sampled_positive_instances.shape[0]])),
            (np.zeros(shape=[sampled_negative_instances.shape[0]])),
        )
    )[order]
    log_array(train_labels, logger, array_name="Train labels")
    log_array(train_data, logger, array_name="Train data")

    return train_data, train_labels


def _validate_and_log_instance_numbers(n_negative, n_positive, n_total_instances):
    n_labeled = n_negative + n_positive
    n_unlabeled = n_total_instances - n_labeled
    missing_msg = "Zero %s found in training data. Positive and negative labels are required to train a model."
    if n_labeled == 0:
        raise ValueError(missing_msg % "labeled instances")
    if n_positive == 0:
        raise ValueError(missing_msg % "positive labeled instances")
    if n_negative == 0:
        raise ValueError(missing_msg % "negative labeled instances")
    logger.info(
        f"Dataset contains {n_labeled} ({round(100 * n_labeled / (n_labeled + n_unlabeled), 2)}%) labeled instances of a total of {n_labeled + n_unlabeled} instances."
    )


def _subsample(instances, sample_size):
    if isinstance(instances, da.Array):
        instances.compute_chunk_sizes()
    n_instances = instances.shape[0]
    indices = np.arange(n_instances)
    sample_indices = np.random.choice(indices, size=min(sample_size, n_instances), replace=False)
    return instances[sample_indices]


def _set_compute_mode(compute_mode: Literal["normal", "parallel", "safe"], chunks: dict) -> dict:
    """Set compute mode for read_input_and_labels_and_save_predictions."""

    # Set default chunks
    if chunks is None:
        chunks = {"band": 1, "x": 1024, "y": 1024}

    # Default dask kwargs
    dask_kwargs = {}

    match compute_mode:
        case "normal":
            return dask_kwargs
        case "parallel":
            dask.config.set(scheduler="threads")
            dask_kwargs["chunks"] = chunks
        case "safe":
            dask.config.set(scheduler="synchronous")
            dask_kwargs["chunks"] = chunks
        case _:
            msg = f"Invalid compute mode: {compute_mode}"
            raise ValueError(msg)
    return dask_kwargs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Classify pixels in the input image using a model trained on the labels."
    )

    parser.add_argument("-i", "--input", type=Path, help="Path to the input TIFF file")
    parser.add_argument("-lp", "--pos_labels", type=Path, help="Path to the positive training labels file, shp or gpkg")
    parser.add_argument("-ln", "--neg_labels", type=Path, help="Path to the negative training labels file, shp or gpkg")
    parser.add_argument("-p", "--predictions", type=Path, help="Path to the predictions output TIFF file")
    parser.add_argument(
        "-f",
        "--feature_type",
        type=str,
        choices=[f.name for f in list(FeatureType)],
        default=FeatureType.FLAIR,
        help=f"Type of feature being used. Default: {FeatureType.FLAIR.name}",
    )
    parser.add_argument(
        "-m",
        "--compute_mode",
        type=str,
        choices=["normal", "parallel", "safe"],
        default="normal",
        help="Mode for reading the input raster data.",
    )
    parser.add_argument(
        "-c",
        "--chunks",
        type=dict,
        default={"band": 1, "x": 1024, "y": 1024},
        help="Chunk size for reading the input raster data.",
    )
    parser.add_argument(
        "-o",
        "--chunk_overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap size for reading the input raster data.",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.exists() or not args.input.is_file():
        parser.error(f"The input file {args.input} does not exist or is not a file.")

    return args


if __name__ == "__main__":
    args = _parse_args()
    input_path = args.input
    pos_labels_path = args.pos_labels
    neg_labels_path = args.neg_labels
    predictions_path = args.predictions
    feature_type = FeatureType.from_string(args.feature_type)
    compute_mode = args.compute_mode
    chunk_overlap = args.chunk_overlap
    chunks = args.chunks

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    read_input_and_labels_and_save_predictions(
        input_path,
        pos_labels_path,
        neg_labels_path,
        predictions_path,
        feature_type=feature_type,
        chunks=chunks,
        chunk_overlap=chunk_overlap,
        compute_mode=compute_mode,
    )
