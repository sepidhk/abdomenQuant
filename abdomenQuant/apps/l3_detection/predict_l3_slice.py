import argparse
import json
import os

from abdomenQuant.dicomseries import DicomSeries
from .model import load_model
from .utils import preprocess_for_detection, rescale_prediction, save_overlay

parser = argparse.ArgumentParser(description='Predict L3 Slice Location From Axial Scan')
parser.add_argument('--dicom-dir', type=str, help='Directory containing the dicom files')
parser.add_argument('--model-path', type=str, help='Path to the model')
parser.add_argument('--model-weights', type=str, help='Path to the model weights')
parser.add_argument('--output-dir', type=str, help='Directory to save the output')
args = parser.parse_args()


def main():
    # Load the dicom series
    dicom_series = DicomSeries(args.dicom_dir, filepattern='*', window_center=30, window_width=150, read_images=True)
    # make sure the output directory exists
    output_dir = f'{args.output_dir}/MRN{dicom_series.mrn}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load the model
    model = load_model(args.model_path, args.model_weights)
    # Preprocess the images
    image_preprocessed, changed_image = preprocess_for_detection(dicom_series.frontal, dicom_series.spacing, 1)
    # Predict the L3 slice location
    prediction = model.predict(image_preprocessed)
    # Change the prediction to original image space
    original_l3 = rescale_prediction(prediction, dicom_series.spacing)
    # Save the output
    save_overlay(dicom_series, original_l3, output_dir)
    json.dump({'l3': original_l3.tolist()}, open(
        output_dir + f'/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_l3_location.json', 'w'))

    return original_l3


if __name__ == '__main__':
    main()
