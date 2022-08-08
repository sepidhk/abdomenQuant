import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage.measure import label, perimeter


# @njit
def _binarize_image(image: np.ndarray, threshold: int = -500) -> np.ndarray:
    """
    A function to binarize a ct scan slice between air and body \n
    :param image: dicom pixel image
    :param threshold: the threshold value to binarize the image (default: -500)
    :return: the binary image as a numpy array
    """
    binary_image = (image.copy() > threshold) * 1
    binary_image = binary_fill_holes(binary_image)
    return binary_image


def _get_largest_connected_component(binary_image: np.ndarray) -> np.ndarray:
    """
    A function to get the largest connected component from binary image (in this case it should be the body)
    :param binary_image: the output from binarize_image
    :return: an numpy array containing only the largest connected component
    """
    labels = label(binary_image)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(int)
    return labels_max


def _remove_exterior_artifacts(ct_image: np.ndarray, largest_connected_component: np.ndarray) -> np.ndarray:
    """
    A function to remove artifacts outside the body from CT scans
    :param ct_image: the original ct image
    :param largest_connected_component: the output from get_largest_connected_component
    :return: the cleaned up original image
    """
    ct_image[largest_connected_component != 1] = np.min(ct_image)
    return ct_image


def _measure_circumference(body_array: np.ndarray, pixel_width: float, pixel_units: str = 'mm'):
    """
    Measure the circumference of the body in centimeters
    :param body_array: the output from mark_body
    :param pixel_width: the width of a pixel in centimeters
    :param pixel_units: the units of the pixel width (default: mm)
    :return: the circumference of the body in centimeters
    """
    waist_pixels = perimeter(body_array, neighbourhood=4)
    waist_cm = np.round(waist_pixels * pixel_width, 2)
    if pixel_units == 'mm':
        waist_cm = waist_cm / 10
    return waist_cm


def get_waist_circumference(axial_array: np.ndarray, l3_slice: np.ndarray, spacing: list) -> float:
    """
    A function to measure the circumference of the body in centimeters
    :param axial_array: the axial ct scan series
    :param l3_slice: the index of the l3 axial slice
    :param spacing: the spacing of the ct scan in millimeters
    :return: the circumference of the body in centimeters
    """
    l3_image = axial_array[l3_slice]
    # Remove exterior artifacts
    binary_l3 = _binarize_image(l3_image)
    body = _get_largest_connected_component(binary_l3)
    l3_image = _remove_exterior_artifacts(l3_image, body)
    # Measure circumference
    l3_image = _binarize_image(l3_image)
    # Erosion is to remove small artifacts attached to body (clips from table, etc
    l3_image = binary_erosion(l3_image, iterations=2)
    waist_circ = _measure_circumference(l3_image, spacing[0])
    return waist_circ
