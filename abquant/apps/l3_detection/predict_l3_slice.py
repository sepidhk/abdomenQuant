import argparse
import json
import os

from abquant.dicomseries import DicomSeries
from .model import load_model
from .utils import preprocess_for_detection, rescale_prediction, save_overlay

parser = argparse.ArgumentParser(description='Predict L3 Slice Location From Axial Scan')
parser.add_argument('--dicom-dir', type=str, help='Directory containing the dicom files')
parser.add_argument('--slice-model-path', type=str, help='Path to the model')
parser.add_argument('--slice-model-weights', type=str, help='Path to the model weights')
parser.add_argument('--output-dir', type=str, help='Directory to save the output')
parser.add_argument('--plot-outputs', type=bool, help='Plot the outputs', default=True)
args = parser.parse_args()


def main(args, dicom_series=None, output_dir=None):
    if dicom_series is None:
        # Load the dicom series
        dicom_series = DicomSeries(args.dicom_dir, filepattern='*', window_center=30, window_width=150, read_images=True)
    if output_dir is None:
        # make sure the output directory exists
        output_dir = f'{args.output_dir}/MRN{dicom_series.mrn}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # Load the model
    model = load_model(args.slice_model_path, args.slice_model_weights)
    # Preprocess the images
    image_preprocessed, changed_image = preprocess_for_detection(dicom_series.frontal, dicom_series.spacing, 1)
    # Predict the L3 slice location
    prediction = model.predict(image_preprocessed)
    # Change the prediction to original image space
    original_l3, probability = rescale_prediction(prediction, dicom_series.spacing)
    if args.plot_outputs:
        # Save the output
        save_overlay(dicom_series, original_l3, output_dir)
    slice_info_dict = {'l3': original_l3.tolist(), 'confidence': probability.tolist()}
    json.dump(slice_info_dict, open(
        output_dir + f'/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_l3_info.json', 'w'))

    return original_l3, slice_info_dict


if __name__ == '__main__':
    main(args)
