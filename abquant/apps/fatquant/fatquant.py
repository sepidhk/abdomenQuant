import numpy as np
import keras.backend as K
from .resnet import build_unet
from scipy.ndimage import binary_fill_holes, binary_erosion
# from abdomenQuant.abquant.dicomseries import DicomSeries
from abdomenQuant.abquant.apps.wcirc.utils import _get_largest_connected_component as lcc
from abdomenQuant.abquant.apps.wcirc.utils import get_waist_circumference
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--dicom-dir', type=str, help='Directory containing the dicom files')
# parser.add_argument('--abdomen-weights', type=str, help='Weights for model to segment the abdominal wall')
# # parser.add_argument('--start', type=int, help='Starting slice from the CT scan')
# # parser.add_argument('--end', type=int, help='Ending slice from the CT scan')
# args = parser.parse_args()


def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


def fatquant(dicom_series=None, start=None, end=None, abdomen_weights=None):
    K.set_image_data_format('channels_last')
    if dicom_series is None:
        # Load the dicom series
        # dicom_series = DicomSeries(dicom_dir, filepattern='*', window_center=30, window_width=150, read_images=True)
        raise ValueError("Need a valid DicomSeries object passed")
    # Set the boundaries
    h, w, d = dicom_series.spacing
    h /= 10
    w /= 10
    start_pt = int(start)
    end_pt = int(end + 1)
    # Extract the abdomen
    abdomen = dicom_series.pixel_array[start_pt: end_pt, :, :].copy()
    abdomen_norm = normalize_image(abdomen)[..., np.newaxis]
    # Build the model
    model = build_unet((512, 512, 1), base_filter=32)
    print(abdomen_weights)
    model.load_weights(abdomen_weights)
    # Predict the abdominal wall
    preds = model.predict(abdomen_norm)
    preds = np.squeeze(preds)
    preds = np.round(preds)

    measures = {}
    # print(start_pt, end_pt)
    for i in range(start_pt, end_pt):
        image = dicom_series.pixel_array[i].copy()
        pred = lcc(preds[i - start_pt])
        pred = binary_fill_holes(pred)
        pred = binary_erosion(pred, iterations=2)
        inside = np.ma.masked_where(pred == 1, image)
        inside = np.ma.getmask(inside)
        outside = np.ma.masked_where(pred == 0, image)
        outside = np.ma.getmask(outside)
        # Get the visceral area
        visceral_im = image.copy()
        visceral_im[inside == False] = np.min(image)
        visceral_pixels = ((visceral_im > -150) & (visceral_im < -30)).sum()
        visceral_area = visceral_pixels * h * w
        # Get the subcutaneous area
        subq_im = image.copy()
        subq_im[outside == False] = np.min(image)
        subq_pixels = (subq_im > -150) & (subq_im < -30)
        subq_pixels = lcc(subq_pixels).sum()
        subq_area = subq_pixels * h * w

        _, wc = get_waist_circumference(dicom_series.pixel_array, i, dicom_series.spacing)
        measures[i] = {
            'waist_circumference': float(wc),
            'visceral_pixels': int(visceral_pixels),
            'visceral_area': float(visceral_area),
            'subq_pixels': int(subq_pixels),
            'subq_area': float(subq_area)
        }

    return measures
